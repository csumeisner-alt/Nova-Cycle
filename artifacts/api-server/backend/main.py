"""
NovaCycle API - Main FastAPI Application
Entry point for the NovaCycle backend service.
Serves all endpoints under the /api prefix on port 8080.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from database.db import create_tables, get_session_factory
from ingestion.pipeline import IngestionPipeline
from routers import predictions, data, history, notifications

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("novacycle")

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
scheduler = AsyncIOScheduler()
pipeline = IngestionPipeline()


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

    # 2. Initialize data pipeline (fetch history if empty, else incremental)
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await pipeline.initialize(session)
        logger.info("Data ingestion pipeline initialized.")
    except Exception as e:
        logger.warning(f"Data initialization warning (will retry on schedule): {e}")

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

    scheduler.start()
    logger.info("APScheduler started.")

    yield  # Application is running

    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("NovaCycle API shut down.")


async def _run_incremental_update():
    """Scheduled task: fetch new 5-min candles since last stored."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await pipeline.run_incremental_update(session)
    except Exception as e:
        logger.error(f"Incremental update failed: {e}")


async def _run_daily_update():
    """Scheduled task: fetch new daily candles."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await pipeline.run_incremental_update(session, timeframe="daily")
    except Exception as e:
        logger.error(f"Daily update failed: {e}")


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
# Include routers — all prefixed with /api
# ---------------------------------------------------------------------------
app.include_router(predictions.router, prefix="/api", tags=["Predictions"])
app.include_router(data.router, prefix="/api", tags=["Data"])
app.include_router(history.router, prefix="/api", tags=["History"])
app.include_router(notifications.router, prefix="/api", tags=["Notifications"])


# ---------------------------------------------------------------------------
# Connectivity check — used by the Android app to verify it can reach the API
# ---------------------------------------------------------------------------
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
