"""
NovaCycle API - Main FastAPI Application
Entry point for the NovaCycle backend service.
Serves all endpoints under the /api prefix on port 8080.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from database.cleanup_state import mark_cleanup_finished, mark_cleanup_started
from database.startup_state import mark_startup_ok, mark_startup_degraded, mark_retrain_skipped
from database.db import create_tables, get_session_factory
from database.maintenance import reclassify_session_labels
from database.ohlc_cleanup import remove_malformed_candles
from ingestion.pipeline import IngestionPipeline
from ml.trainer import ModelTrainer
from ml.training_status import record_training_result
from routers import predictions, data, history, notifications, releases

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("novacycle")

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
scheduler = AsyncIOScheduler()
pipeline = IngestionPipeline()
trainer = ModelTrainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: create DB tables, initialize data, start scheduler.
    Shutdown: stop scheduler gracefully.
    """
    logger.info("NovaCycle API starting up...")

    # 1. Create database tables
    await create_tables()
    logger.info("Database tables ready.")

    # 2. Initialize data pipeline in the background so the API starts fast.
    #    The committed SQLite DB already contains historical data; this task
    #    only needs to catch up incremental updates. Endpoints gracefully
    #    return "no data" messages if the DB is temporarily empty.
    async def _init_pipeline():
        # Safety net: the initialized guard MUST be released before this
        # coroutine exits regardless of how it exits.  pipeline.initialize()
        # releases it in its own finally block, but if anything before that
        # call raises (e.g. get_session_factory, reclassify_session_labels)
        # the except below would swallow the error and return — leaving
        # every scheduled job blocked forever on wait_for_initialized().
        init_succeeded = False
        try:
            try:
                session_factory = get_session_factory()
                async with session_factory() as session:
                    corrected = await reclassify_session_labels(session)
                    if corrected:
                        logger.info(
                            f"Repaired {corrected} candle(s) with stale session labels."
                        )
                    await pipeline.initialize(session)
                    await session.commit()
                logger.info("Data ingestion pipeline initialized.")
                mark_startup_ok()
                init_succeeded = True
            except Exception as e:
                logger.warning(f"Data initialization warning (will retry on schedule): {e}")
                mark_startup_degraded(str(e))

            # Startup retrain check: catch up if models are stale (>7 days) or missing.
            # Skipped when initialization failed — retraining on a partially-ingested
            # or empty DB could produce a degraded model that gets persisted and served.
            if init_succeeded:
                await _run_weekly_retrain()
            else:
                logger.warning(
                    "Startup retrain skipped: initialization did not complete cleanly."
                )
                mark_retrain_skipped()
        finally:
            # Guarantee: release the initialized guard even if an unhandled
            # exception escapes the inner try blocks above (e.g. the session
            # factory itself raising before initialize() is ever entered).
            # pipeline.initialize() already sets the flag in its own finally,
            # so calling set() again here is always idempotent.
            pipeline._initialized_flag = True
            pipeline._get_initialized_event().set()

    asyncio.create_task(_init_pipeline())

    # 2b. One-time retroactive cleanup: remove candles stored before ingest-time
    #     OHLC validation was in place.  Launched as its own background task so
    #     it never delays _init_pipeline() or the first request.  A hard 300 s
    #     timeout prevents it from running indefinitely on very large databases.
    asyncio.create_task(_run_startup_cleanup())

    # 3. Configure APScheduler for incremental updates
    # Every 5 minutes during extended-hours trading window: Mon-Fri 04:00-20:00 ET
    scheduler.add_job(
        _run_incremental_update,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour="4-20",
            minute="*/5",
            timezone="America/New_York"
        ),
        id="incremental_5min",
        name="5-min incremental update",
        replace_existing=True,
        misfire_grace_time=60
    )

    # Daily candle update after market close (16:30 ET)
    scheduler.add_job(
        _run_daily_update,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=16,
            minute=35,
            timezone="America/New_York"
        ),
        id="daily_update",
        name="Daily candle update",
        replace_existing=True
    )

    # Weekly retrain check — Sundays 03:00 ET (market closed). retrain_if_needed()
    # is a no-op when models were trained within the last 7 days, so running it
    # both at startup and weekly is safe.
    scheduler.add_job(
        _run_weekly_retrain,
        trigger=CronTrigger(
            day_of_week="sun",
            hour=3,
            minute=0,
            timezone="America/New_York"
        ),
        id="weekly_retrain",
        name="Weekly model retrain check",
        replace_existing=True,
        misfire_grace_time=3600
    )

    scheduler.start()
    logger.info("APScheduler started.")

    yield  # Application is running

    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("NovaCycle API shut down.")


async def _run_incremental_update():
    """Scheduled task: fetch new 5-min candles since last stored.

    Waits for the startup pipeline to finish before running so that the
    scheduler firing in the brief window between scheduler.start() and
    initialize() completing doesn't race with cleanup.
    """
    await pipeline.wait_for_initialized()
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await pipeline.run_incremental_update(session)
            await session.commit()
    except Exception as e:
        logger.error(f"Incremental update failed: {e}")


async def _run_daily_update():
    """Scheduled task: fetch new daily candles.

    Waits for the startup pipeline to finish before running (same guard as
    _run_incremental_update).
    """
    await pipeline.wait_for_initialized()
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await pipeline.run_incremental_update(session)
            await session.commit()
    except Exception as e:
        logger.error(f"Daily update failed: {e}")


async def _run_weekly_retrain():
    """Scheduled task: retrain models if they are older than the weekly threshold.

    Failures are recorded in the training-status file so /api/healthz reports
    a degraded training state instead of failing silently.
    """
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            retrained = await trainer.retrain_if_needed(session)
            await session.commit()
        if retrained:
            logger.info("Weekly retrain check: retraining performed.")
        else:
            logger.info("Weekly retrain check: models up-to-date.")
    except Exception as e:
        logger.error(f"Weekly retrain failed: {e}")
        # Surface the failure through the training-status health flag.
        for model_name in ("long_trend", "short_trend"):
            record_training_result(
                model_name, success=False, error=f"Weekly retrain job failed: {e}"
            )


#: Hard time-limit for the background OHLC cleanup task (seconds).
#: Override with CLEANUP_TIMEOUT_SECONDS environment variable.
_CLEANUP_TIMEOUT_SECONDS = float(os.environ.get("CLEANUP_TIMEOUT_SECONDS", "300"))


async def _run_startup_cleanup():
    """Background task: scan all candle tables and delete malformed rows.

    Runs independently of _init_pipeline() so the server starts accepting
    requests immediately.  A hard timeout (default 300 s) ensures it cannot
    block the event loop indefinitely on very large databases.

    State is exposed through database.cleanup_state so /api/healthz can report
    a ``cleanup_pending`` flag while the task is in progress.
    """
    mark_cleanup_started()
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            try:
                summary = await asyncio.wait_for(
                    remove_malformed_candles(session),
                    timeout=_CLEANUP_TIMEOUT_SECONDS,
                )
                await session.commit()
                if summary["rows_found"]:
                    logger.info(
                        "ohlc_startup_cleanup rows_found=%d rows_removed=%d "
                        "tables=%s timeframes=%s",
                        summary["rows_found"], summary["rows_removed"],
                        summary["tables_affected"], summary["timeframes_affected"],
                    )
                else:
                    logger.info("ohlc_startup_cleanup: no malformed candles found.")
            except asyncio.TimeoutError:
                logger.warning(
                    "ohlc_startup_cleanup: timed out after %.0f s — "
                    "cleanup aborted; will retry on next startup.",
                    _CLEANUP_TIMEOUT_SECONDS,
                )
                await session.rollback()
    except Exception as exc:
        logger.error("ohlc_startup_cleanup failed (non-fatal): %s", exc)
    finally:
        mark_cleanup_finished()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NovaCycle API",
    description=(
        "AI-powered VOO ETF trading signal system. "
        "Provides long-trend and short-trend signals, confidence history, "
        "and signal explanations. "
        "NOTE: Model currently trained only for ticker='VOO'. "
        "Multi-ticker support will be added later."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Structured error handling — the Android app never receives a bare 500/HTML
# error page; every failure mode maps to {"ok": false, "error": <CODE>, ...}.
# Successful responses are untouched, so existing client models keep parsing.
# ---------------------------------------------------------------------------

#: Per-request processing budget in seconds. Generous enough for cold-start
#: model loads; override with REQUEST_TIMEOUT_SECONDS. Docs endpoints excluded.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))

_TIMEOUT_EXEMPT_PATHS = ("/docs", "/redoc", "/openapi.json")


@app.middleware("http")
async def request_timeout_middleware(request: Request, call_next):
    """Bound request processing time and return a structured TIMEOUT error."""
    if request.url.path.startswith(_TIMEOUT_EXEMPT_PATHS):
        return await call_next(request)
    try:
        return await asyncio.wait_for(
            call_next(request), timeout=REQUEST_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error(
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s: "
            f"{request.method} {request.url.path}"
        )
        # This middleware runs OUTSIDE CORSMiddleware (added later via decorator,
        # so it wraps the stack), meaning the 504 would otherwise ship without
        # CORS headers and browser clients couldn't read the structured error.
        # Mirror the app's permissive CORS policy manually.
        headers = {}
        if request.headers.get("origin"):
            headers["Access-Control-Allow-Origin"] = "*"
        return JSONResponse(
            status_code=504,
            content={
                "ok": False,
                "error": "TIMEOUT",
                "detail": f"Request exceeded {REQUEST_TIMEOUT_SECONDS:.0f}s processing budget",
            },
            headers=headers,
        )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Malformed query params / body → 422 INVALID_REQUEST."""
    return JSONResponse(
        status_code=422,
        content={"ok": False, "error": "INVALID_REQUEST", "detail": exc.errors()},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Keep intentional HTTPException semantics, add the structured envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": "INVALID_REQUEST" if exc.status_code < 500 else "SERVER_FAILURE",
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Any uncaught error → 500 SERVER_FAILURE (never a bare traceback page)."""
    logger.exception(
        f"Unhandled error on {request.method} {request.url.path}: {exc}"
    )
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "SERVER_FAILURE", "detail": str(exc)},
    )

# ---------------------------------------------------------------------------
# Include routers — all prefixed with /api
# ---------------------------------------------------------------------------
app.include_router(predictions.router, prefix="/api", tags=["Predictions"])
app.include_router(data.router, prefix="/api", tags=["Data"])
app.include_router(history.router, prefix="/api", tags=["History"])
app.include_router(notifications.router, prefix="/api", tags=["Notifications"])
app.include_router(releases.router, prefix="/api", tags=["Releases"])


# ---------------------------------------------------------------------------
# Connectivity check — used by the Android app to verify it can reach the API
# ---------------------------------------------------------------------------
APK_DOWNLOAD_PATH = (
    Path(__file__).resolve().parent / "downloads" / "novacycle-latest.apk"
)


@app.get("/api/downloads/novacycle-latest.apk", include_in_schema=False)
async def download_latest_apk():
    """Download the locally built release APK without requiring GitHub."""
    return FileResponse(
        APK_DOWNLOAD_PATH,
        media_type="application/vnd.android.package-archive",
        filename="novacycle-latest.apk",
    )


@app.get("/api")
async def api_root():
    """Return a lightweight readiness response for the artifact proxy."""
    return {
        "service": "NovaCycle API",
        "status": "running",
        "health": "/api/healthz",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/test")
async def api_test():
    return {
        "status": "ok",
        "service": "NovaCycle API",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "NovaCycle API",
        "status": "running",
        "docs": "/docs",
        "api_prefix": "/api",
        "timestamp": datetime.utcnow().isoformat()
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
