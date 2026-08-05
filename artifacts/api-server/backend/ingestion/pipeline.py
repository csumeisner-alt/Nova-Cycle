"""
NovaCycle Ingestion Pipeline
=============================
Orchestrates data fetching, gap/session tagging, and DB storage.

Schedule:
  - Every 5 min during market hours (09:25 – 20:05 ET Mon–Fri)
  - Daily after close for daily candles

NOTE: "Pipeline currently fetches only VOO. Multi-ticker ingestion will be added later."
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.models import (
    VooCandle, VixCandle, SpxCandle,
    VixShortCandle, VixLongCandle, RatesCandle,
    CreditHyCandle, CreditIgCandle, BreadthCandle,
)
from ingestion import market_calendar
from ingestion.fetcher import DataFetcher, ohlc_validation_issue

logger = logging.getLogger(__name__)

# Maximum seconds to wait for a single yfinance backfill range fetch.
# A hung vendor call must not delay the next scheduled ingestion tick.
BACKFILL_FETCH_TIMEOUT_SECS: float = 30.0

# ─────────────────────────────────────────────────────────────────────────────
# Last 5-min stall recovery attempt (module-level so /healthz can report it
# regardless of which pipeline instance ran the recovery).
#   {"last_attempt_at": iso|None, "outcome": "recovered"|"failed"|"skipped_cooldown"|None,
#    "bars_fetched": int|None}
# ─────────────────────────────────────────────────────────────────────────────
_last_5min_recovery_status: dict = {
    "last_attempt_at": None,
    "outcome": None,
    "bars_fetched": None,
}


def get_5min_recovery_status() -> dict:
    """Return the last 5-min stall recovery attempt summary for /healthz.

    Uses the in-memory record when one exists (an attempt happened this
    process lifetime); otherwise falls back to the persisted record so a
    backend restart does not hide evidence that a recovery fired or failed
    shortly before the restart. Always includes the persisted rolling
    history and cumulative failure count. Never raises.
    """
    status = dict(_last_5min_recovery_status)
    try:
        from ingestion.recovery_history import get_persisted_recovery_status

        persisted = get_persisted_recovery_status()
        if status.get("last_attempt_at") is None and persisted["last_attempt"]:
            last = persisted["last_attempt"]
            status["last_attempt_at"] = last.get("last_attempt_at")
            status["outcome"] = last.get("outcome")
            status["bars_fetched"] = last.get("bars_fetched")
            status["from_previous_run"] = True
        else:
            status["from_previous_run"] = False
        status["history"] = persisted["history"]
        status["failure_count"] = persisted["failure_count"]
    except Exception as exc:
        logger.error("recovery_status persisted lookup failed: %s", exc)
        status.setdefault("from_previous_run", False)
        status.setdefault("history", [])
        status.setdefault("failure_count", 0)
    return status


def _record_5min_recovery(outcome: str, at: datetime, bars_fetched: Optional[int]) -> None:
    _last_5min_recovery_status["last_attempt_at"] = at.isoformat()
    _last_5min_recovery_status["outcome"] = outcome
    _last_5min_recovery_status["bars_fetched"] = bars_fetched
    # Persist so the record (and rolling history / failure count) survives
    # a backend restart. Never raises.
    try:
        from ingestion.recovery_history import record_recovery_attempt

        record_recovery_attempt(outcome, at.isoformat(), bars_fetched)
    except Exception as exc:
        logger.error("recovery_status persist failed: %s", exc)


class IngestionPipeline:
    """Manages full and incremental ingestion of VOO and VIX market data."""

    def __init__(self):
        self.fetcher = DataFetcher()
        # Timestamp of the last 5-min stall recovery attempt (cooldown guard).
        self._last_5min_recovery_attempt: Optional[datetime] = None
        # Set to True once initialize() has completed so scheduled jobs can
        # check whether startup is done without importing asyncio at call sites.
        # The matching asyncio.Event is created lazily (inside the running loop)
        # to avoid "attached to a different loop" issues in tests.
        self._initialized_flag: bool = False
        self._initialized_event: Optional[asyncio.Event] = None
        # Tracks which context feed_keys were stale on the previous incremental
        # run so we only fire an alert on the *transition* (healthy→stale),
        # not on every subsequent stale check.
        self._prev_stale_context_feeds: set[str] = set()

    # ─────────────────────────────────────────────────────────────────────────
    # Startup-ordering guard
    # ─────────────────────────────────────────────────────────────────────────

    def _get_initialized_event(self) -> asyncio.Event:
        """Return (creating lazily) the asyncio.Event that guards scheduled jobs.

        The event is created on first call so it is always attached to the
        event loop that is actually running — safe both in production and in
        pytest-asyncio tests, each of which creates a fresh loop.
        """
        if self._initialized_event is None:
            self._initialized_event = asyncio.Event()
            if self._initialized_flag:
                # initialize() already completed before the event was created
                # (e.g. during a test that bypasses the normal startup path).
                self._initialized_event.set()
        return self._initialized_event

    async def wait_for_initialized(self) -> None:
        """Await until initialize() has completed.

        Called by every scheduled job so that a job firing in the brief window
        between scheduler.start() and initialize() completing will simply pause
        rather than run on a partially-cleaned DB.
        """
        await self._get_initialized_event().wait()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self, db_session: AsyncSession) -> None:
        """
        Smart initialisation:
          - If no VOO daily data in DB → run full historical fetch + store
          - If data exists → run incremental update only

        Always sets the _initialized guard on exit (success or failure is
        handled by the caller; the guard prevents scheduled jobs from waiting
        forever if initialize() raises).
        """
        logger.info("IngestionPipeline.initialize() called")
        try:
            await self.remove_invalid_voo_candles(db_session)
            await self.remove_invalid_vix_candles(db_session)
            await self.remove_invalid_spx_candles(db_session)

            # Check whether we already have daily candles
            result = await db_session.execute(
                select(func.count(VooCandle.id)).where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == "daily",
                )
            )
            count = result.scalar() or 0

            if count == 0:
                logger.info("No existing data found. Running full historical fetch…")
                await self._run_full_fetch(db_session)
            else:
                logger.info(
                    "Found %d daily candles in DB. Running incremental update…", count
                )
                await self.run_incremental_update(db_session)

            # A previously successful deployment may have VOO history but
            # only a short VIX history.  Incremental ingestion alone cannot
            # repair that because it starts at the latest stored VIX date.
            # Reconcile the historical VIX start against VOO after normal
            # startup ingestion, without blocking the main feed if Yahoo is
            # unavailable.
            await self._ensure_vix_history_coverage(db_session)
        finally:
            # Signal scheduled jobs that startup is done regardless of outcome.
            self._initialized_flag = True
            self._get_initialized_event().set()
            logger.info("IngestionPipeline initialized — scheduled jobs may now run.")

    async def _ensure_vix_history_coverage(self, db_session: AsyncSession) -> None:
        """Backfill VIX when its stored range starts materially after VOO.

        VIX is an index, so zero volume remains valid.  This method is
        deliberately best-effort: predictions can continue with an explicit
        missing-data feature if the vendor is unavailable.
        """
        try:
            voo_result = await db_session.execute(
                select(func.min(VooCandle.timestamp)).where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == "daily",
                )
            )
            vix_result = await db_session.execute(
                select(func.min(VixCandle.timestamp)).where(
                    VixCandle.ticker == settings.VIX_TICKER,
                    VixCandle.timeframe == "daily",
                )
            )
            voo_dates_result = await db_session.execute(
                select(VooCandle.timestamp).where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == "daily",
                    VooCandle.is_extended_hours == False,
                ).order_by(VooCandle.timestamp)
            )
            vix_dates_result = await db_session.execute(
                select(VixCandle.timestamp).where(
                    VixCandle.ticker == settings.VIX_TICKER,
                    VixCandle.timeframe == "daily",
                )
            )
            voo_start = voo_result.scalar()
            vix_start = vix_result.scalar()
            if voo_start is None:
                return
            voo_dates = {row[0].date() for row in voo_dates_result.fetchall()}
            vix_dates = {row[0].date() for row in vix_dates_result.fetchall()}
            missing_dates = sorted(voo_dates - vix_dates)
            if not missing_dates:
                return

            start = datetime.combine(missing_dates[0], datetime.min.time())
            end = datetime.combine(missing_dates[-1], datetime.min.time())
            logger.warning(
                "vix_history_coverage_gap voo_start=%s vix_start=%s "
                "missing_days=%d action=backfill",
                voo_start.date().isoformat(),
                vix_start.date().isoformat() if vix_start else None,
                len(missing_dates),
            )
            df = await self.fetcher.fetch_vix_daily_range(start, end)
            if df.empty:
                logger.warning("vix_history_coverage_backfill_empty")
                return
            await self.store_vix_candles(
                df, db_session, timeframe="daily", _is_backfill=True
            )
            logger.info(
                "vix_history_coverage_backfill_complete rows=%d", len(df)
            )
        except Exception as exc:
            logger.error("vix_history_coverage_backfill_failed error=%s", exc)

    async def remove_invalid_voo_candles(self, db_session: AsyncSession) -> int:
        """Remove malformed VOO rows left by older ingestion versions.

        New fetches are validated before storage, but a bad row may already
        exist from a prior process lifetime.  Remove it before predictions
        select their feature window; the next incremental fetch re-reads the
        boundary date and stores it only if the vendor returns a valid candle.

        Checks removed:
          - OHLC consistency violations (high < open, low > close, etc.)
          - Zero-volume bars (volume == 0 or NULL) — yfinance glitch days
        """
        result = await db_session.execute(
            select(VooCandle).where(VooCandle.ticker == settings.TICKER)
        )

        def _row_issue(row: VooCandle) -> str | None:
            """Return a reason string if *row* is invalid, else None."""
            ohlc_issue = ohlc_validation_issue(row.open, row.high, row.low, row.close)
            if ohlc_issue:
                return ohlc_issue
            vol = row.volume
            if vol is None or float(vol) == 0:
                return "zero_volume"
            return None

        invalid_rows = [
            (row, _row_issue(row))
            for row in result.scalars().all()
            if _row_issue(row) is not None
        ]
        if not invalid_rows:
            return 0

        for row, issue in invalid_rows:
            logger.warning(
                "ingest_invalid_existing_candle_removed timeframe=%s ts=%s issue=%s",
                row.timeframe,
                row.timestamp.isoformat(),
                issue,
            )
            await db_session.delete(row)
        await db_session.flush()
        logger.warning(
            "ingest_invalid_existing_candles_removed count=%d",
            len(invalid_rows),
        )
        return len(invalid_rows)

    async def remove_invalid_vix_candles(self, db_session: AsyncSession) -> int:
        """Remove malformed VIX rows left by older ingestion versions.

        ``^VIX`` is an index, not a traded security. Yahoo Finance normally
        reports valid OHLC values with volume 0, so volume is not a validity
        gate for VIX.
        """
        result = await db_session.execute(
            select(VixCandle).where(VixCandle.ticker == settings.VIX_TICKER)
        )

        invalid_rows = [
            (row, ohlc_validation_issue(row.open, row.high, row.low, row.close))
            for row in result.scalars().all()
        ]
        invalid_rows = [(row, issue) for row, issue in invalid_rows if issue]
        if not invalid_rows:
            return 0

        for row, issue in invalid_rows:
            logger.warning(
                "ingest_invalid_existing_candle_removed ticker=%s timeframe=%s ts=%s issue=%s",
                settings.VIX_TICKER,
                row.timeframe,
                row.timestamp.isoformat(),
                issue,
            )
            await db_session.delete(row)
        await db_session.flush()
        logger.warning(
            "ingest_invalid_existing_candles_removed ticker=%s count=%d",
            settings.VIX_TICKER,
            len(invalid_rows),
        )
        return len(invalid_rows)

    async def remove_invalid_spx_candles(self, db_session: AsyncSession) -> int:
        """Remove zero-volume SPX rows left by older ingestion versions.

        New fetches are gated by the zero-volume check in store_spx_candles,
        but a glitch row may already exist in the database from a prior process
        lifetime.  Remove it at startup so predictions never read it.

        Checks removed:
          - Zero-volume bars (volume == 0 or NULL) — yfinance glitch days
        """
        result = await db_session.execute(
            select(SpxCandle).where(SpxCandle.ticker == settings.SPX_FUTURES_TICKER)
        )

        invalid_rows = [
            row
            for row in result.scalars().all()
            if row.volume is None or float(row.volume) == 0
        ]
        if not invalid_rows:
            return 0

        for row in invalid_rows:
            logger.warning(
                "ingest_invalid_existing_candle_removed ticker=%s timeframe=%s ts=%s issue=zero_volume",
                settings.SPX_FUTURES_TICKER,
                row.timeframe,
                row.timestamp.isoformat(),
            )
            await db_session.delete(row)
        await db_session.flush()
        logger.warning(
            "ingest_invalid_existing_candles_removed ticker=%s count=%d",
            settings.SPX_FUTURES_TICKER,
            len(invalid_rows),
        )
        return len(invalid_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Full historical fetch
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_full_fetch(self, db_session: AsyncSession) -> None:
        """Fetch full HISTORY_YEARS of VOO + VIX and persist to DB."""

        # ── VOO ──────────────────────────────────────────────────────────────
        data = await self.fetcher.fetch_historical_voo(years=settings.HISTORY_YEARS)
        daily_df = data.get("daily", pd.DataFrame())
        fivemin_df = data.get("5min", pd.DataFrame())

        if not daily_df.empty:
            await self.store_voo_candles(daily_df, db_session, timeframe="daily")
        if not fivemin_df.empty:
            await self.store_voo_candles(fivemin_df, db_session, timeframe="5min")

        # ── VIX ──────────────────────────────────────────────────────────────
        vix_df = await self.fetcher.fetch_historical_vix(years=settings.HISTORY_YEARS)
        if not vix_df.empty:
            await self.store_vix_candles(vix_df, db_session, timeframe="daily")

        # ── SPX futures ───────────────────────────────────────────────────────
        spx_df = await self.fetcher.fetch_historical_spx(years=settings.HISTORY_YEARS)
        if not spx_df.empty:
            await self.store_spx_candles(spx_df, db_session, timeframe="daily")

        # ── Broader market context (VIX9D, VIX3M, TNX, HYG, LQD, NYAD) ──────
        await self._ingest_context_tickers(db_session, years=settings.HISTORY_YEARS)

        logger.info("Full historical fetch complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Incremental update
    # ─────────────────────────────────────────────────────────────────────────

    async def run_incremental_update(self, db_session: AsyncSession) -> None:
        """
        Fetch only candles newer than the latest timestamp already in DB.
        Appends new rows; never re-fetches full history.
        """

        # ── Find the last daily timestamp ─────────────────────────────────────
        result = await db_session.execute(
            select(func.max(VooCandle.timestamp)).where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "daily",
            )
        )
        last_daily: Optional[datetime] = result.scalar()

        # ── Find the last 5-min timestamp ─────────────────────────────────────
        result = await db_session.execute(
            select(func.max(VooCandle.timestamp)).where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "5min",
            )
        )
        last_5min: Optional[datetime] = result.scalar()

        # Use the oldest of the two (or epoch if none)
        last_ts = last_daily or datetime(2000, 1, 1, tzinfo=timezone.utc)

        logger.info("Incremental update: last_daily=%s, last_5min=%s", last_daily, last_5min)

        data = await self.fetcher.fetch_incremental_voo(last_timestamp=last_ts)

        daily_df = data.get("daily", pd.DataFrame())
        fivemin_df = data.get("5min", pd.DataFrame())

        if not daily_df.empty:
            await self.store_voo_candles(daily_df, db_session, timeframe="daily")
        if not fivemin_df.empty:
            await self.store_voo_candles(fivemin_df, db_session, timeframe="5min")

        # ── Incremental VIX ───────────────────────────────────────────────────
        try:
            vix_df = await self.fetcher.fetch_historical_vix(years=1)
            if not vix_df.empty:
                # Store the full fetched window (duplicates are skipped) so
                # missing-day detection can heal downtime holes older than
                # the last stored VIX timestamp.
                await self.store_vix_candles(vix_df, db_session, timeframe="daily")
        except Exception as exc:
            # VIX is an optional macro input. A vendor failure must preserve
            # the existing neutral/fallback prediction behavior and be surfaced
            # by the staleness check below.
            logger.error("vix_incremental_fetch_failed error=%s", exc)

        # ── Incremental SPX futures ───────────────────────────────────────────
        result = await db_session.execute(
            select(func.max(SpxCandle.timestamp)).where(
                SpxCandle.ticker == settings.SPX_FUTURES_TICKER
            )
        )
        last_spx: Optional[datetime] = result.scalar()

        # Fetch a rolling window (full history when never ingested) — duplicates
        # are skipped, so downtime holes heal automatically.
        spx_years = 1 if last_spx else settings.HISTORY_YEARS
        spx_df = await self.fetcher.fetch_historical_spx(years=spx_years)
        if not spx_df.empty:
            await self.store_spx_candles(spx_df, db_session, timeframe="daily")

        # ── Broader market context (VIX9D, VIX3M, TNX, HYG, LQD, NYAD) ──────
        await self._ingest_context_tickers(db_session, years=1)

        # ── Broader context staleness check ───────────────────────────────────
        try:
            ctx_results = await check_context_staleness(db_session)
            # Alert for feeds that are *newly* stale (healthy last run, stale now).
            newly_stale = [
                s for s in ctx_results
                if s.get("stale")
                and s.get("feed_key") not in self._prev_stale_context_feeds
            ]
            if newly_stale:
                asyncio.create_task(
                    _notify_context_stale_feeds(newly_stale),
                    name="notify_context_stale_feeds",
                )
            # Update tracker: stale feed_keys as of this run.
            self._prev_stale_context_feeds = {
                s["feed_key"] for s in ctx_results if s.get("stale")
            }
        except Exception as exc:
            logger.error("context_staleness_check_failed error=%s", exc)

        # ── SPX staleness check ───────────────────────────────────────────────
        # If yfinance quietly stops returning ES=F data, the macro signal
        # silently degrades to the VOO proxy. Surface it loudly here.
        try:
            await check_spx_staleness(db_session)
        except Exception as exc:
            logger.error("spx_staleness_check_failed error=%s", exc)

        # ── VIX staleness check ───────────────────────────────────────────────
        # If yfinance quietly stops returning ^VIX data, the macro sensitivity
        # signal silently degrades. Surface it loudly here.
        try:
            await check_vix_staleness(db_session)
        except Exception as exc:
            logger.error("vix_staleness_check_failed error=%s", exc)

        # ── Daily candle feed staleness check ─────────────────────────────────
        # If the ingestion pipeline silently stops writing daily candles, the
        # weekly retrain won't notice for up to 7 days.  Surface it now.
        try:
            await check_daily_candle_staleness(db_session)
        except Exception as exc:
            logger.error("daily_candle_staleness_check_failed error=%s", exc)

        # ── VOO 5-min staleness check ─────────────────────────────────────────
        # If yfinance quietly stops returning intraday bars, the short-trend
        # signal silently goes stale during market hours. Surface it loudly.
        try:
            fivemin_status = await check_5min_staleness(db_session)
            if fivemin_status.get("stale"):
                # ── Immediate targeted re-fetch on stall detection ────────────
                # One-shot recovery attempt (cooldown-guarded, never a tight
                # loop); recovery failures never abort the ingestion run.
                try:
                    await self.recover_5min_stall(fivemin_status, db_session)
                except Exception as exc:
                    logger.error("fivemin_stall_recovery_error error=%s", exc)
        except Exception as exc:
            logger.error("fivemin_staleness_check_failed error=%s", exc)

        logger.info("Incremental update complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # 5-min stall recovery (immediate targeted re-fetch on staleness)
    # ─────────────────────────────────────────────────────────────────────────

    # Minimum minutes between recovery attempts so repeated stale runs (the
    # scheduler fires every 5 min) can't turn into a re-fetch loop.
    FIVEMIN_RECOVERY_COOLDOWN_MINUTES: int = 15

    async def recover_5min_stall(
        self,
        status: dict,
        db_session: AsyncSession,
        now: Optional[datetime] = None,
    ) -> dict:
        """
        One-shot targeted re-fetch of the missing intraday window after
        check_5min_staleness flips to stale.

        Behaviour:
          - Cooldown-guarded (FIVEMIN_RECOVERY_COOLDOWN_MINUTES) so back-to-back
            stale scheduler runs don't hammer yfinance; skipped attempts log
            `fivemin_stall_recovery_skipped`.
          - Fetches only the gap window: from the last stored 5-min bar (or
            today's session when none exists) through now via fetch_5min_range.
          - Re-runs the staleness check afterwards and logs the outcome
            distinctly: `fivemin_stall_recovered` (INFO) on success,
            `fivemin_stall_recovery_failed` (WARNING) on continued staleness.

        Returns a summary dict:
            {"attempted": bool, "recovered": Optional[bool],
             "bars_fetched": int, "reason": Optional[str]}
        """
        if now is None:
            now = datetime.utcnow()
        if now.tzinfo is not None:
            now = now.astimezone(timezone.utc).replace(tzinfo=None)

        summary: dict = {
            "attempted": False, "recovered": None,
            "bars_fetched": 0, "reason": None,
        }

        last = getattr(self, "_last_5min_recovery_attempt", None)
        cooldown = self.FIVEMIN_RECOVERY_COOLDOWN_MINUTES
        if last is not None:
            since = (now - last).total_seconds() / 60.0
            if since < cooldown:
                summary["reason"] = "cooldown"
                _record_5min_recovery("skipped_cooldown", now, None)
                logger.info(
                    "fivemin_stall_recovery_skipped reason=cooldown "
                    "minutes_since_last=%.1f cooldown=%d",
                    since, cooldown,
                )
                return summary

        self._last_5min_recovery_attempt = now
        summary["attempted"] = True

        # Gap window: from the last stored bar's day (or today) through now.
        latest_iso = status.get("latest_5min")
        if latest_iso:
            try:
                start = datetime.fromisoformat(latest_iso)
            except ValueError:
                start = now
        else:
            start = now

        logger.warning(
            "fivemin_stall_recovery_attempt window=%s→%s age_minutes=%s",
            start.date().isoformat(), now.date().isoformat(),
            status.get("age_minutes"),
        )

        df = await self.fetcher.fetch_5min_range(start, now)
        if not df.empty and latest_iso:
            # Keep only bars newer than what we already have.
            df = df[df.index > pd.Timestamp(start)]
        summary["bars_fetched"] = int(len(df))

        if not df.empty:
            await self.store_voo_candles(
                df, db_session, timeframe="5min", _is_backfill=True
            )

        recheck = await check_5min_staleness(db_session, now=now)
        summary["recovered"] = not recheck.get("stale", True)
        _record_5min_recovery(
            "recovered" if summary["recovered"] else "failed",
            now, summary["bars_fetched"],
        )

        if summary["recovered"]:
            logger.info(
                "fivemin_stall_recovered bars_fetched=%d latest_5min=%s",
                summary["bars_fetched"], recheck.get("latest_5min"),
            )
        else:
            logger.warning(
                "fivemin_stall_recovery_failed bars_fetched=%d latest_5min=%s "
                "age_minutes=%s — feed still stale after targeted re-fetch; "
                "next attempt after cooldown.",
                summary["bars_fetched"], recheck.get("latest_5min"),
                recheck.get("age_minutes"),
            )
        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # Scheduled updates (called by APScheduler)
    # ─────────────────────────────────────────────────────────────────────────

    async def run_scheduled_updates(self, db_session: AsyncSession) -> None:
        """
        APScheduler callback:
          - 5-min during market hours (incl. extended): fetch latest 5-min candles
          - Triggered separately daily after close for daily candles
        Called from main.py scheduler.
        """
        logger.info("Scheduled ingestion update triggered at %s", datetime.utcnow())
        try:
            await self.run_incremental_update(db_session)
        except Exception as exc:
            logger.error("Scheduled update failed: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Storage helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def store_voo_candles(
        self,
        candles: pd.DataFrame,
        db_session: AsyncSession,
        timeframe: str,
        _is_backfill: bool = False,
    ) -> None:
        """
        Persist VOO candles to DB.

        Steps:
          1. Compute gap and session type for each candle
          2. Skip duplicates (same ticker + timestamp + timeframe)
          3. Bulk-insert new rows

        Gap formula:
          GapPercent = (PreMarketOpen - PreviousClose) / PreviousClose * 100
        """
        if candles.empty:
            return

        ticker = settings.TICKER
        inserted = 0
        repaired = 0
        skipped = 0

        # ── In-frame duplicate detection ──────────────────────────────────────
        # yfinance occasionally returns the same bar twice; keep the last
        # occurrence and log the anomaly instead of inserting duplicates.
        # Note: even if this in-frame check fails, the per-row
        # existing_timestamps guard below still prevents duplicate inserts,
        # because each inserted timestamp is added to the set.
        try:
            if not isinstance(candles.index, pd.DatetimeIndex):
                candles = candles.copy()
                candles.index = pd.to_datetime(candles.index)
            dup_count = int(candles.index.duplicated(keep="last").sum())
            if dup_count > 0:
                candles = candles[~candles.index.duplicated(keep="last")]
                logger.warning(
                    "ingest_duplicate_candles timeframe=%s dropped=%d",
                    timeframe, dup_count,
                )
        except Exception as exc:
            logger.error("ingest_duplicate_check_failed error=%s", exc)

        # Pre-load existing timestamps to avoid duplicate queries
        result = await db_session.execute(
            select(VooCandle).where(
                VooCandle.ticker == ticker,
                VooCandle.timeframe == timeframe,
            )
        )
        existing_rows = result.scalars().all()
        existing_by_timestamp = {row.timestamp: row for row in existing_rows}
        existing_timestamps = set(existing_by_timestamp)

        # ── Missing-candle detection + targeted backfill (daily + 5min) ───────
        # Detect trading days with no candle in the covered window, then
        # proactively re-fetch just those date ranges. Backfill failures are
        # logged and never abort the regular ingestion run.
        missing_days: list = []
        try:
            if timeframe in ("daily", "5min") and len(candles) >= 1 and not _is_backfill:
                idx = candles.sort_index().index
                have = {ts.date() for ts in idx}
                win_start = idx[0].date()
                win_end = idx[-1].date()

                if timeframe == "5min":
                    # Also count sessions already in the DB, so gaps caused
                    # by downtime (days present in neither the DB nor this
                    # fetched frame) are detected. Clamp the window to
                    # yfinance's ~60-day 5-min fetch limit and never look
                    # earlier than the oldest 5-min session we know about.
                    db_days = {ts.date() for ts in existing_timestamps}
                    have |= db_days
                    fetch_floor = (
                        datetime.now(timezone.utc).date() - timedelta(days=58)
                    )
                    earliest_known = min(have) if have else win_start
                    win_start = max(earliest_known, fetch_floor)

                if win_start <= win_end:
                    missing_days = [
                        d.date()
                        for d in pd.date_range(win_start, win_end, freq="D")
                        if market_calendar.is_trading_day(d.date())
                        and d.date() not in have
                    ]
                if missing_days:
                    logger.warning(
                        "ingest_missing_candles timeframe=%s count=%d days=%s",
                        timeframe,
                        len(missing_days),
                        ",".join(d.isoformat() for d in missing_days[:20]),
                    )
        except Exception as exc:
            logger.error("ingest_missing_check_failed error=%s", exc)
            missing_days = []

        # Iterate sorted by time (oldest first for gap calc)
        prev_close: Optional[float] = None

        for ts, row in candles.sort_index().iterrows():
            # Normalise timestamp
            ts_naive = ts.to_pydatetime()
            if ts_naive.tzinfo is not None:
                ts_naive = ts_naive.replace(tzinfo=None)

            try:
                open_price = float(row.get("open", 0.0))
                high_price = float(row.get("high", 0.0))
                low_price = float(row.get("low", 0.0))
                close_price = float(row.get("close", 0.0))
                volume = float(row.get("volume", 0.0))
            except (TypeError, ValueError):
                logger.warning(
                    "ingest_ohlc_anomaly issue=non_numeric_ohlc timeframe=%s ts=%s",
                    timeframe, ts_naive.isoformat(),
                )
                skipped += 1
                continue

            issue = ohlc_validation_issue(
                open_price, high_price, low_price, close_price
            )
            if issue is not None:
                logger.warning(
                    "ingest_ohlc_anomaly issue=%s timeframe=%s ts=%s",
                    issue, timeframe, ts_naive.isoformat()
                )
                skipped += 1
                continue

            if volume == 0:
                logger.warning(
                    "ingest_zero_volume_bar_skipped timeframe=%s ts=%s",
                    timeframe, ts_naive.isoformat(),
                )
                skipped += 1
                continue

            existing = existing_by_timestamp.get(ts_naive)
            if existing is not None:
                # Valid duplicate rows remain immutable.  A previously stored
                # malformed row is the exception: replace it with the newly
                # validated vendor row so the repair is retroactive.
                old_issue = ohlc_validation_issue(
                    existing.open, existing.high, existing.low, existing.close
                )
                if old_issue is None:
                    skipped += 1
                    continue

                existing.open = open_price
                existing.high = high_price
                existing.low = low_price
                existing.close = close_price
                existing.volume = volume
                existing.is_extended_hours = bool(row.get("is_extended_hours", False))
                existing.session_type = str(row.get("session_type", "regular"))
                repaired += 1
                logger.warning(
                    "ingest_ohlc_repaired timeframe=%s ts=%s old_issue=%s",
                    timeframe, ts_naive.isoformat(), old_issue
                )
                continue

            # Session classification
            is_extended = bool(row.get("is_extended_hours", False))
            session_type = str(row.get("session_type", "regular"))

            # Gap detection (only for the first pre-market candle of each day)
            gap_percent = 0.0
            gap_type = "none"

            if session_type == "pre_market" and prev_close is not None:
                open_price = float(row.get("open", 0.0))
                # Pass the same day's regular-session candles (if already in
                # this frame) so detect_gap can compute real gap_momentum.
                # gap_momentum is additive and not persisted (no schema
                # change); it is logged here and recomputed at read time by
                # /api/gap_status.
                post_open = None
                try:
                    if "session_type" in candles.columns:
                        same_day = candles.index.normalize() == pd.Timestamp(ts_naive).normalize()
                        post_open = candles[same_day & (candles["session_type"] == "regular")]
                except Exception:
                    post_open = None
                gap_info = await self.fetcher.detect_gap(
                    prev_close, open_price, post_open_candles=post_open
                )
                gap_percent = gap_info["gap_percent"]
                gap_type = gap_info["gap_type"]
                if gap_info.get("gap_momentum") is not None:
                    logger.info(
                        "gap_momentum ts=%s gap_percent=%.4f momentum=%.4f",
                        ts_naive.isoformat(), gap_percent, gap_info["gap_momentum"],
                    )

            # Update prev_close for regular-session closing bars
            if session_type == "regular":
                prev_close = float(row.get("close", prev_close or 0.0))

            candle = VooCandle(
                ticker=ticker,
                timestamp=ts_naive,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timeframe=timeframe,
                is_extended_hours=is_extended,
                session_type=session_type,
                gap_percent=gap_percent,
                gap_type=gap_type,
            )
            db_session.add(candle)
            existing_timestamps.add(ts_naive)
            inserted += 1

        await db_session.flush()
        logger.info(
            "VOO %s candles: inserted=%d, repaired=%d, skipped=%d",
            timeframe, inserted, repaired, skipped,
        )

        # ── Targeted backfill of missing trading days ─────────────────────────
        if missing_days:
            try:
                await self._backfill_missing_days(
                    missing_days, db_session, timeframe=timeframe
                )
            except Exception as exc:
                logger.error("ingest_backfill_failed timeframe=%s error=%s", timeframe, exc)

    @staticmethod
    def _group_contiguous_days(days: list) -> list[tuple]:
        """Group a sorted list of dates into (start, end) ranges where
        consecutive entries are at most 4 calendar days apart (so weekends
        and short holidays don't split a range)."""
        from datetime import timedelta as _td

        ranges: list[tuple] = []
        days = sorted(days)
        start = prev = days[0]
        for d in days[1:]:
            if (d - prev).days <= 4:
                prev = d
            else:
                ranges.append((start, prev))
                start = prev = d
        ranges.append((start, prev))
        return ranges

    async def _backfill_missing_days(
        self,
        missing_days: list,
        db_session: AsyncSession,
        timeframe: str = "daily",
    ) -> None:
        """
        Re-fetch and store candles for the given missing trading days, for
        either the daily or 5-min timeframe.

        Failures are logged per range and never propagate to the caller's
        regular ingestion flow.
        """
        from datetime import datetime as _dt, time as _time

        ranges = self._group_contiguous_days(missing_days)
        logger.info(
            "ingest_backfill_start timeframe=%s days=%d ranges=%d",
            timeframe, len(missing_days), len(ranges),
        )

        filled = 0
        for start_d, end_d in ranges:
            try:
                start = _dt.combine(start_d, _time.min)
                end = _dt.combine(end_d, _time.min)
                if timeframe == "5min":
                    df = await asyncio.wait_for(
                        self.fetcher.fetch_5min_range(start, end),
                        timeout=BACKFILL_FETCH_TIMEOUT_SECS,
                    )
                else:
                    df = await asyncio.wait_for(
                        self.fetcher.fetch_daily_range(start, end),
                        timeout=BACKFILL_FETCH_TIMEOUT_SECS,
                    )
                if df.empty:
                    logger.warning(
                        "ingest_backfill_empty timeframe=%s range=%s→%s",
                        timeframe, start_d.isoformat(), end_d.isoformat(),
                    )
                    continue
                await self.store_voo_candles(
                    df, db_session, timeframe=timeframe, _is_backfill=True
                )
                filled += 1
            except asyncio.TimeoutError:
                logger.error(
                    "ingest_backfill_range_timeout timeframe=%s range=%s→%s "
                    "timeout=%.0fs",
                    timeframe, start_d.isoformat(), end_d.isoformat(),
                    BACKFILL_FETCH_TIMEOUT_SECS,
                )
            except Exception as exc:
                logger.error(
                    "ingest_backfill_range_failed timeframe=%s range=%s→%s error=%s",
                    timeframe, start_d.isoformat(), end_d.isoformat(), exc,
                )

        logger.info(
            "ingest_backfill_complete timeframe=%s ranges_ok=%d ranges_total=%d",
            timeframe, filled, len(ranges),
        )

    async def store_vix_candles(
        self,
        candles: pd.DataFrame,
        db_session: AsyncSession,
        timeframe: str,
        _is_backfill: bool = False,
    ) -> None:
        """
        Persist VIX candles to DB, skipping duplicates.

        For daily candles, missing trading days inside the covered window are
        detected (same trading-day calendar logic as VOO) and re-fetched via a
        targeted backfill. Backfill failures are logged and never abort the
        main run.
        """
        if candles.empty:
            return

        ticker = settings.VIX_TICKER
        inserted = 0
        skipped = 0

        result = await db_session.execute(
            select(VixCandle.timestamp).where(
                VixCandle.ticker == ticker,
                VixCandle.timeframe == timeframe,
            )
        )
        existing_timestamps = set(row[0] for row in result.fetchall())

        # ── Missing-candle detection (daily only) ─────────────────────────────
        missing_days: list = []
        try:
            if timeframe == "daily" and len(candles) >= 1 and not _is_backfill:
                if not isinstance(candles.index, pd.DatetimeIndex):
                    candles = candles.copy()
                    candles.index = pd.to_datetime(candles.index)
                idx = candles.sort_index().index
                have = {ts.date() for ts in idx}
                # Include days already in the DB so downtime holes (days in
                # neither the DB nor this frame) are detected within the window.
                have |= {ts.date() for ts in existing_timestamps}
                win_start = idx[0].date()
                win_end = idx[-1].date()

                if win_start <= win_end:
                    missing_days = [
                        d.date()
                        for d in pd.date_range(win_start, win_end, freq="D")
                        if market_calendar.is_trading_day(d.date())
                        and d.date() not in have
                    ]
                if missing_days:
                    logger.warning(
                        "vix_ingest_missing_candles count=%d days=%s",
                        len(missing_days),
                        ",".join(d.isoformat() for d in missing_days[:20]),
                    )
        except Exception as exc:
            logger.error("vix_ingest_missing_check_failed error=%s", exc)
            missing_days = []

        for ts, row in candles.sort_index().iterrows():
            ts_naive = ts.to_pydatetime()
            if ts_naive.tzinfo is not None:
                ts_naive = ts_naive.replace(tzinfo=None)

            if ts_naive in existing_timestamps:
                skipped += 1
                continue

            try:
                open_price = float(row.get("open", 0.0))
                high_price = float(row.get("high", 0.0))
                low_price = float(row.get("low", 0.0))
                close_price = float(row.get("close", 0.0))
                raw_volume = row.get("volume", 0.0)
                volume = 0.0 if pd.isna(raw_volume) else float(raw_volume)
            except (TypeError, ValueError):
                logger.warning(
                    "ingest_ohlc_anomaly ticker=%s timeframe=%s ts=%s issue=non_numeric",
                    ticker, timeframe, ts_naive.isoformat(),
                )
                skipped += 1
                continue

            issue = ohlc_validation_issue(
                open_price, high_price, low_price, close_price
            )
            if issue or volume < 0:
                logger.warning(
                    "ingest_ohlc_anomaly ticker=%s timeframe=%s ts=%s issue=%s",
                    ticker,
                    timeframe,
                    ts_naive.isoformat(),
                    issue or "negative_volume",
                )
                skipped += 1
                continue

            candle = VixCandle(
                ticker=ticker,
                timestamp=ts_naive,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timeframe=timeframe,
            )
            db_session.add(candle)
            existing_timestamps.add(ts_naive)
            inserted += 1

        await db_session.flush()
        logger.info(
            "VIX %s candles: inserted=%d, skipped=%d (duplicates)",
            timeframe, inserted, skipped,
        )

        # ── Targeted backfill of missing VIX trading days ─────────────────────
        if missing_days:
            try:
                await self._backfill_missing_vix_days(missing_days, db_session)
            except Exception as exc:
                logger.error("vix_ingest_backfill_failed error=%s", exc)

    async def store_spx_candles(
        self,
        candles: pd.DataFrame,
        db_session: AsyncSession,
        timeframe: str = "daily",
    ) -> None:
        """
        Persist SPX futures candles to DB, skipping duplicates.

        Simpler than the VIX path (no missing-day backfill): the macro
        sensitivity feature forward-fills over gaps, and incremental runs
        re-fetch a rolling window that heals holes automatically.
        """
        if candles.empty:
            return

        ticker = settings.SPX_FUTURES_TICKER
        inserted = 0
        skipped = 0

        result = await db_session.execute(
            select(SpxCandle.timestamp).where(
                SpxCandle.ticker == ticker,
                SpxCandle.timeframe == timeframe,
            )
        )
        existing_timestamps = set(row[0] for row in result.fetchall())

        for ts, row in candles.sort_index().iterrows():
            ts_naive = ts.to_pydatetime()
            if ts_naive.tzinfo is not None:
                ts_naive = ts_naive.replace(tzinfo=None)

            if ts_naive in existing_timestamps:
                skipped += 1
                continue

            volume = float(row.get("volume", 0.0))
            if volume == 0:
                logger.warning(
                    "ingest_zero_volume_bar_skipped ticker=%s timeframe=%s ts=%s",
                    ticker, timeframe, ts_naive.isoformat(),
                )
                skipped += 1
                continue

            candle = SpxCandle(
                ticker=ticker,
                timestamp=ts_naive,
                open=float(row.get("open", 0.0)),
                high=float(row.get("high", 0.0)),
                low=float(row.get("low", 0.0)),
                close=float(row.get("close", 0.0)),
                volume=volume,
                timeframe=timeframe,
            )
            db_session.add(candle)
            existing_timestamps.add(ts_naive)
            inserted += 1

        await db_session.flush()
        logger.info(
            "SPX %s candles: inserted=%d, skipped=%d (duplicates)",
            timeframe, inserted, skipped,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Broader market context ingestion
    # ─────────────────────────────────────────────────────────────────────────

    # Mapping: config ticker → (ORM model, label, is_index)
    # is_index=True  → zero-volume bars are allowed (^VIX9D, ^VIX3M, ^TNX, ^NYAD)
    # is_index=False → zero-volume bars filtered (HYG, LQD are ETFs)
    _CONTEXT_SOURCES = [
        # (attr_name_on_settings, model_class, label, is_index)
        ("VIX_SHORT_TICKER",  VixShortCandle,  "VIX9D",  True),
        ("VIX_LONG_TICKER",   VixLongCandle,   "VIX3M",  True),
        ("RATES_TICKER",      RatesCandle,     "TNX",    True),
        ("CREDIT_HY_TICKER",  CreditHyCandle,  "HYG",    False),
        ("CREDIT_IG_TICKER",  CreditIgCandle,  "LQD",    False),
        ("BREADTH_TICKER",    BreadthCandle,   "NYAD",   True),
    ]

    async def _ingest_context_tickers(
        self,
        db_session: AsyncSession,
        years: int = 10,
    ) -> None:
        """Fetch and store daily candles for all six broader-context tickers.

        Each source is fetched independently; a single vendor failure for one
        ticker never aborts the others.  Empty tickers (settings value "")
        are silently skipped.
        """
        for attr, model, label, is_index in self._CONTEXT_SOURCES:
            ticker = getattr(settings, attr, "")
            if not ticker:
                logger.debug(
                    "context_ingest_skipped label=%s reason=empty_ticker", label
                )
                continue
            try:
                df = await self.fetcher.fetch_historical_context_ticker(
                    ticker, years=years, is_index=is_index
                )
                if not df.empty:
                    await self.store_context_candles(
                        df, db_session,
                        model=model, ticker=ticker, label=label,
                        is_index=is_index,
                    )
                else:
                    logger.warning(
                        "context_ingest_empty label=%s ticker=%s", label, ticker
                    )
            except Exception as exc:
                logger.error(
                    "context_ingest_failed label=%s ticker=%s error=%s",
                    label, ticker, exc,
                )

    async def store_context_candles(
        self,
        candles: pd.DataFrame,
        db_session: AsyncSession,
        *,
        model,
        ticker: str,
        label: str,
        is_index: bool = True,
        _is_backfill: bool = False,
    ) -> None:
        """
        Persist daily broader-context candles to DB, skipping duplicates.

        Generic helper shared by all six context feeds (VIX9D, VIX3M, TNX,
        HYG, LQD, NYAD).  Missing trading days inside the covered window are
        detected (same trading-day calendar logic as VOO and VIX) and
        re-fetched via a targeted backfill.  Backfill failures are logged and
        never abort the main run.

        Args:
            model:       SQLAlchemy ORM class (e.g. VixShortCandle).
            ticker:      The canonical ticker string stored in the model's
                         ticker column (matches the config setting).
            label:       Short human-readable name for log messages.
            is_index:    When True, zero-volume bars are accepted (indices
                         such as ^VIX9D report volume 0 from Yahoo).
            _is_backfill: When True, skip the gap-detection pass so backfill
                          frames never trigger a recursive second round.
        """
        if candles.empty:
            return

        inserted = 0
        skipped = 0

        result = await db_session.execute(
            select(model.timestamp).where(
                model.ticker == ticker,
                model.timeframe == "daily",
            )
        )
        existing_timestamps = set(row[0] for row in result.fetchall())

        # ── Missing-candle detection (daily only, non-backfill) ───────────────
        # Detect trading days with no candle inside the fetched frame window.
        # Days already in the DB are merged into `have` so that downtime holes
        # (days in neither the DB nor this frame) are also detected.
        missing_days: list = []
        try:
            if not _is_backfill and len(candles) >= 1:
                if not isinstance(candles.index, pd.DatetimeIndex):
                    candles = candles.copy()
                    candles.index = pd.to_datetime(candles.index)
                idx = candles.sort_index().index
                have = {ts.date() for ts in idx}
                # Include days already in the DB so outage holes (days absent
                # from both the DB and this fetched frame) are caught.
                have |= {ts.date() for ts in existing_timestamps}
                win_start = idx[0].date()
                win_end = idx[-1].date()

                if win_start <= win_end:
                    missing_days = [
                        d.date()
                        for d in pd.date_range(win_start, win_end, freq="D")
                        if market_calendar.is_trading_day(d.date())
                        and d.date() not in have
                    ]
                if missing_days:
                    logger.warning(
                        "context_ingest_missing_candles label=%s count=%d days=%s",
                        label,
                        len(missing_days),
                        ",".join(d.isoformat() for d in missing_days[:20]),
                    )
        except Exception as exc:
            logger.error(
                "context_ingest_missing_check_failed label=%s error=%s", label, exc
            )
            missing_days = []

        for ts, row in candles.sort_index().iterrows():
            ts_naive = ts.to_pydatetime()
            if ts_naive.tzinfo is not None:
                ts_naive = ts_naive.replace(tzinfo=None)

            if ts_naive in existing_timestamps:
                skipped += 1
                continue

            try:
                open_p  = float(row.get("open",  0.0))
                high_p  = float(row.get("high",  0.0))
                low_p   = float(row.get("low",   0.0))
                close_p = float(row.get("close", 0.0))
                raw_vol = row.get("volume", 0.0)
                volume  = 0.0 if pd.isna(raw_vol) else float(raw_vol)
            except (TypeError, ValueError):
                logger.warning(
                    "ingest_ohlc_anomaly ticker=%s ts=%s issue=non_numeric",
                    ticker, ts_naive.isoformat(),
                )
                skipped += 1
                continue

            issue = ohlc_validation_issue(open_p, high_p, low_p, close_p)
            if issue or volume < 0:
                logger.warning(
                    "ingest_ohlc_anomaly ticker=%s ts=%s issue=%s",
                    ticker, ts_naive.isoformat(), issue or "negative_volume",
                )
                skipped += 1
                continue

            candle = model(
                ticker=ticker,
                timestamp=ts_naive,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=volume,
                timeframe="daily",
            )
            db_session.add(candle)
            existing_timestamps.add(ts_naive)
            inserted += 1

        await db_session.flush()
        logger.info(
            "%s daily candles: inserted=%d, skipped=%d (duplicates)",
            label, inserted, skipped,
        )

        # ── Targeted backfill of missing context trading days ─────────────────
        if missing_days:
            try:
                await self._backfill_missing_context_days(
                    missing_days, db_session,
                    model=model, ticker=ticker, label=label, is_index=is_index,
                )
            except Exception as exc:
                logger.error(
                    "context_ingest_backfill_failed label=%s error=%s", label, exc
                )

    async def _backfill_missing_context_days(
        self,
        missing_days: list,
        db_session: AsyncSession,
        *,
        model,
        ticker: str,
        label: str,
        is_index: bool = True,
    ) -> None:
        """
        Re-fetch and store daily candles for the given missing trading days
        for a single broader-context feed (VIX9D, VIX3M, TNX, HYG, LQD, or
        NYAD).  Failures are logged per range and never propagate to the
        caller's regular ingestion flow.
        """
        from datetime import datetime as _dt, time as _time

        ranges = self._group_contiguous_days(missing_days)
        logger.info(
            "context_ingest_backfill_start label=%s days=%d ranges=%d",
            label, len(missing_days), len(ranges),
        )

        filled = 0
        for start_d, end_d in ranges:
            try:
                start = _dt.combine(start_d, _time.min)
                end = _dt.combine(end_d, _time.min)
                df = await asyncio.wait_for(
                    self.fetcher.fetch_context_ticker_range(
                        ticker, start, end, is_index=is_index
                    ),
                    timeout=BACKFILL_FETCH_TIMEOUT_SECS,
                )
                if df.empty:
                    logger.warning(
                        "context_ingest_backfill_empty label=%s range=%s→%s",
                        label, start_d.isoformat(), end_d.isoformat(),
                    )
                    continue
                await self.store_context_candles(
                    df, db_session,
                    model=model, ticker=ticker, label=label,
                    is_index=is_index, _is_backfill=True,
                )
                filled += 1
            except asyncio.TimeoutError:
                logger.error(
                    "context_ingest_backfill_range_timeout label=%s "
                    "range=%s→%s timeout=%.0fs",
                    label, start_d.isoformat(), end_d.isoformat(),
                    BACKFILL_FETCH_TIMEOUT_SECS,
                )
            except Exception as exc:
                logger.error(
                    "context_ingest_backfill_range_failed label=%s "
                    "range=%s→%s error=%s",
                    label, start_d.isoformat(), end_d.isoformat(), exc,
                )

        logger.info(
            "context_ingest_backfill_complete label=%s ranges_ok=%d ranges_total=%d",
            label, filled, len(ranges),
        )

    async def _backfill_missing_vix_days(
        self,
        missing_days: list,
        db_session: AsyncSession,
    ) -> None:
        """
        Re-fetch and store daily VIX candles for the given missing trading
        days. Failures are logged per range and never propagate to the
        caller's regular ingestion flow.
        """
        from datetime import datetime as _dt, time as _time

        ranges = self._group_contiguous_days(missing_days)
        logger.info(
            "vix_ingest_backfill_start days=%d ranges=%d",
            len(missing_days), len(ranges),
        )

        filled = 0
        for start_d, end_d in ranges:
            try:
                start = _dt.combine(start_d, _time.min)
                end = _dt.combine(end_d, _time.min)
                df = await asyncio.wait_for(
                    self.fetcher.fetch_vix_daily_range(start, end),
                    timeout=BACKFILL_FETCH_TIMEOUT_SECS,
                )
                if df.empty:
                    logger.warning(
                        "vix_ingest_backfill_empty range=%s→%s",
                        start_d.isoformat(), end_d.isoformat(),
                    )
                    continue
                await self.store_vix_candles(
                    df, db_session, timeframe="daily", _is_backfill=True
                )
                filled += 1
            except asyncio.TimeoutError:
                logger.error(
                    "vix_ingest_backfill_range_timeout range=%s→%s timeout=%.0fs",
                    start_d.isoformat(), end_d.isoformat(),
                    BACKFILL_FETCH_TIMEOUT_SECS,
                )
            except Exception as exc:
                logger.error(
                    "vix_ingest_backfill_range_failed range=%s→%s error=%s",
                    start_d.isoformat(), end_d.isoformat(), exc,
                )

        logger.info(
            "vix_ingest_backfill_complete ranges_ok=%d ranges_total=%d",
            filled, len(ranges),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Generic feed staleness check (module-level: shared by ingestion + healthz)
# ─────────────────────────────────────────────────────────────────────────────

async def _check_feed_staleness(
    db_session: AsyncSession,
    *,
    model,
    ticker: str,
    max_lag: int,
    feed_key: str,
    missing_detail: str,
    lag_detail: str,
    log_event: str,
) -> dict:
    """
    Generic quiet-data-stop detector for a daily candle feed.

    Compares the latest stored daily candle of `model`/`ticker` against the
    latest VOO daily trading day. When the feed candle lags by more than
    `max_lag` *trading days* (or is missing entirely while VOO data exists),
    the feed has silently gone stale.

    Logs a WARNING (event name `log_event`) when stale and returns a
    structured dict for the health endpoint:

        {
            "stale": bool,
            "latest_<feed_key>": Optional[str],  # ISO date
            "latest_voo": Optional[str],         # ISO date
            "lag_trading_days": Optional[int],
            "max_lag_trading_days": int,
            "detail": Optional[str],             # set when stale
        }

    `missing_detail` is used verbatim when no feed candles exist; `lag_detail`
    is a format string receiving latest_feed, latest_voo, lag, and max_lag.
    """
    result = await db_session.execute(
        select(func.max(model.timestamp)).where(
            model.ticker == ticker,
            model.timeframe == "daily",
        )
    )
    latest_feed: Optional[datetime] = result.scalar()

    result = await db_session.execute(
        select(func.max(VooCandle.timestamp)).where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == "daily",
        )
    )
    latest_voo: Optional[datetime] = result.scalar()

    latest_key = f"latest_{feed_key}"
    status: dict = {
        "stale": False,
        latest_key: latest_feed.date().isoformat() if latest_feed else None,
        "latest_voo": latest_voo.date().isoformat() if latest_voo else None,
        "lag_trading_days": None,
        "max_lag_trading_days": max_lag,
        "detail": None,
    }

    if latest_voo is None:
        # No VOO reference data yet — nothing meaningful to compare against.
        return status

    if latest_feed is None:
        status["stale"] = True
        status["detail"] = missing_detail
        logger.warning(
            "%s latest_%s=none latest_voo=%s — %s",
            log_event, feed_key, status["latest_voo"], status["detail"],
        )
        return status

    # Count trading days in (latest_feed, latest_voo]
    lag = 0
    if latest_feed.date() < latest_voo.date():
        for d in pd.date_range(
            latest_feed.date() + timedelta(days=1), latest_voo.date(), freq="D"
        ):
            if market_calendar.is_trading_day(d.date()):
                lag += 1
    status["lag_trading_days"] = lag

    if lag > max_lag:
        status["stale"] = True
        status["detail"] = lag_detail.format(
            latest_feed=status[latest_key],
            latest_voo=status["latest_voo"],
            lag=lag,
            max_lag=max_lag,
        )
        logger.warning(
            "%s latest_%s=%s latest_voo=%s lag_trading_days=%d max=%d — %s",
            log_event, feed_key, status[latest_key], status["latest_voo"],
            lag, max_lag, status["detail"],
        )
    return status


async def check_context_staleness(db_session: AsyncSession) -> list[dict]:
    """
    Detect quietly-stale broader-context feeds (VIX9D, VIX3M, TNX, HYG, LQD, NYAD).

    Each source is compared against the latest VOO trading day using
    _check_feed_staleness.  Returns a list of one status dict per source;
    stale feeds log a WARNING.  Never raises.
    """
    from config import settings as _s

    sources = [
        dict(
            model=VixShortCandle,
            ticker=_s.VIX_SHORT_TICKER,
            feed_key="vix_short",
            missing_detail=(
                "No VIX9D candles stored while VOO data exists — "
                "vix_term_ratio context feature will fire missing=1.0."
            ),
            lag_detail=(
                "Latest VIX9D candle ({latest_feed}) lags the latest "
                "VOO trading day ({latest_voo}) by {lag} trading days "
                "(max {max_lag}) — vix_term_ratio context feature degraded."
            ),
            log_event="vix_short_data_stale",
        ),
        dict(
            model=VixLongCandle,
            ticker=_s.VIX_LONG_TICKER,
            feed_key="vix_long",
            missing_detail=(
                "No VIX3M candles stored while VOO data exists — "
                "vix_term_ratio context feature will fire missing=1.0."
            ),
            lag_detail=(
                "Latest VIX3M candle ({latest_feed}) lags the latest "
                "VOO trading day ({latest_voo}) by {lag} trading days "
                "(max {max_lag}) — vix_term_ratio context feature degraded."
            ),
            log_event="vix_long_data_stale",
        ),
        dict(
            model=RatesCandle,
            ticker=_s.RATES_TICKER,
            feed_key="rates",
            missing_detail=(
                "No TNX candles stored while VOO data exists — "
                "rates_change context feature will fire missing=1.0."
            ),
            lag_detail=(
                "Latest TNX candle ({latest_feed}) lags the latest "
                "VOO trading day ({latest_voo}) by {lag} trading days "
                "(max {max_lag}) — rates_change context feature degraded."
            ),
            log_event="rates_data_stale",
        ),
        dict(
            model=CreditHyCandle,
            ticker=_s.CREDIT_HY_TICKER,
            feed_key="credit_hy",
            missing_detail=(
                "No HYG candles stored while VOO data exists — "
                "credit_spread context feature will fire missing=1.0."
            ),
            lag_detail=(
                "Latest HYG candle ({latest_feed}) lags the latest "
                "VOO trading day ({latest_voo}) by {lag} trading days "
                "(max {max_lag}) — credit_spread context feature degraded."
            ),
            log_event="credit_hy_data_stale",
        ),
        dict(
            model=CreditIgCandle,
            ticker=_s.CREDIT_IG_TICKER,
            feed_key="credit_ig",
            missing_detail=(
                "No LQD candles stored while VOO data exists — "
                "credit_spread context feature will fire missing=1.0."
            ),
            lag_detail=(
                "Latest LQD candle ({latest_feed}) lags the latest "
                "VOO trading day ({latest_voo}) by {lag} trading days "
                "(max {max_lag}) — credit_spread context feature degraded."
            ),
            log_event="credit_ig_data_stale",
        ),
        dict(
            model=BreadthCandle,
            ticker=_s.BREADTH_TICKER,
            feed_key="breadth",
            missing_detail=(
                "No NYAD candles stored while VOO data exists — "
                "breadth context feature will fire missing=1.0."
            ),
            lag_detail=(
                "Latest NYAD candle ({latest_feed}) lags the latest "
                "VOO trading day ({latest_voo}) by {lag} trading days "
                "(max {max_lag}) — breadth context feature degraded."
            ),
            log_event="breadth_data_stale",
        ),
    ]

    results = []
    for src in sources:
        try:
            status = await _check_feed_staleness(
                db_session,
                model=src["model"],
                ticker=src["ticker"],
                max_lag=_s.LONG_CONTEXT_STALENESS_MAX_DAYS,
                feed_key=src["feed_key"],
                missing_detail=src["missing_detail"],
                lag_detail=src["lag_detail"],
                log_event=src["log_event"],
            )
            # Enrich with human-readable ticker and feed_key for the health endpoint.
            status["ticker"] = src["ticker"]
            status["feed_key"] = src["feed_key"]
            results.append(status)
        except Exception as exc:
            logger.error(
                "context_staleness_check_failed ticker=%s error=%s",
                src["ticker"], exc,
            )
    return results


# Map feed_key → model feature that degrades when the feed goes stale.
# Used to populate the push-notification body so operators know the impact.
_CONTEXT_FEED_FEATURE_MAP: dict[str, str] = {
    "vix_short":  "vix_term_ratio",
    "vix_long":   "vix_term_ratio",
    "rates":      "rates_change",
    "credit_hy":  "credit_spread",
    "credit_ig":  "credit_spread",
    "breadth":    "breadth",
}


async def _notify_context_stale_feeds(stale_feeds: list[dict]) -> None:
    """
    Background task: send push notifications to all registered devices for each
    context feed that has *newly* gone stale (i.e. was healthy on the previous
    check).

    One notification per feed per device is sent.  Errors are logged but never
    raised — this must not interrupt the ingestion run.
    """
    from notifications.fcm import FCMNotifier
    from database.db import get_session_factory
    from database.models import DeviceToken
    from sqlalchemy import select

    if not stale_feeds:
        return

    try:
        session_factory = get_session_factory()
        async with session_factory() as db:
            result = await db.execute(select(DeviceToken))
            tokens = [row.token for row in result.scalars().all()]
    except Exception as exc:
        logger.error("context_stale_alert_token_fetch_failed error=%s", exc)
        return

    if not tokens:
        logger.debug(
            "context_stale_alert_no_tokens feeds=%s",
            [f["feed_key"] for f in stale_feeds],
        )
        return

    notifier = FCMNotifier()
    for feed in stale_feeds:
        ticker = feed.get("ticker") or feed.get("feed_key", "unknown")
        feed_key = feed.get("feed_key", "unknown")
        lag = feed.get("lag_trading_days")
        feature = _CONTEXT_FEED_FEATURE_MAP.get(feed_key, feed_key)

        for token in tokens:
            try:
                sent = await notifier.send_context_stale_alert(
                    device_token=token,
                    ticker=ticker,
                    feed_key=feed_key,
                    lag_trading_days=lag,
                    degraded_feature=feature,
                )
                if sent:
                    logger.info(
                        "context_stale_alert_sent feed=%s ticker=%s lag=%s token=%.20s…",
                        feed_key, ticker, lag, token,
                    )
            except Exception as exc:
                logger.error(
                    "context_stale_alert_send_failed feed=%s error=%s", feed_key, exc
                )


async def check_spx_staleness(db_session: AsyncSession) -> dict:
    """
    Detect quietly-stale SPX futures data (see _check_feed_staleness).

    When SPX lags VOO by more than settings.SPX_STALENESS_MAX_LAG_DAYS
    trading days (or is missing while VOO data exists), the macro signal has
    silently degraded to the VOO overnight proxy.
    """
    return await _check_feed_staleness(
        db_session,
        model=SpxCandle,
        ticker=settings.SPX_FUTURES_TICKER,
        max_lag=settings.SPX_STALENESS_MAX_LAG_DAYS,
        feed_key="spx",
        missing_detail=(
            "No SPX futures candles stored while VOO data exists — "
            "macro signal is running on the VOO overnight proxy."
        ),
        lag_detail=(
            "Latest SPX futures candle ({latest_feed}) lags the "
            "latest VOO trading day ({latest_voo}) by {lag} "
            "trading days (max {max_lag}) — macro signal has degraded to "
            "the VOO overnight proxy."
        ),
        log_event="spx_data_stale",
    )


async def check_vix_staleness(db_session: AsyncSession) -> dict:
    """
    Detect quietly-stale VIX data (see _check_feed_staleness).

    When VIX lags VOO by more than settings.VIX_STALENESS_MAX_LAG_DAYS
    trading days (or is missing while VOO data exists), the macro sensitivity
    signal has silently degraded.
    """
    return await _check_feed_staleness(
        db_session,
        model=VixCandle,
        ticker=settings.VIX_TICKER,
        max_lag=settings.VIX_STALENESS_MAX_LAG_DAYS,
        feed_key="vix",
        missing_detail=(
            "No VIX candles stored while VOO data exists — "
            "the macro sensitivity signal has silently degraded."
        ),
        lag_detail=(
            "Latest VIX candle ({latest_feed}) lags the latest "
            "VOO trading day ({latest_voo}) by {lag} trading "
            "days (max {max_lag}) — the macro sensitivity signal has "
            "degraded."
        ),
        log_event="vix_data_stale",
    )


async def check_daily_candle_staleness(
    db_session: AsyncSession,
    now: Optional[datetime] = None,
) -> dict:
    """
    Detect when the daily VOO candle feed has silently stopped writing new rows.

    Counts trading days between the most recent regular-hours daily VOO candle
    and today (wall-clock UTC). When the gap exceeds
    ``settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS``, the ingestion pipeline has
    likely stopped and this is surfaced as a structured ERROR-level log and a
    ``stale=True`` flag in the health response.

    Returns:

        {
            "stale": bool,
            "latest_daily": Optional[str],       # ISO date of most recent candle
            "lag_trading_days": Optional[int],
            "threshold_trading_days": int,
            "detail": Optional[str],             # set when stale
        }

    ``now`` is injectable for tests; defaults to current UTC (naive, matching
    the DB's UTC-naive timestamps).
    """
    if now is None:
        now = datetime.utcnow()
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)

    threshold = settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS

    result = await db_session.execute(
        select(func.max(VooCandle.timestamp)).where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == "daily",
            VooCandle.is_extended_hours == False,  # noqa: E712
        )
    )
    latest: Optional[datetime] = result.scalar()

    status: dict = {
        "stale": False,
        "latest_daily": latest.date().isoformat() if latest else None,
        "lag_trading_days": None,
        "threshold_trading_days": threshold,
        "detail": None,
    }

    if latest is None:
        # No candles at all — only stale if we are past the startup grace window;
        # a fresh deployment with no data yet is not the same as a stopped feed.
        # We leave stale=False here; the retrain no-data error covers this case.
        return status

    # Count trading days in (latest.date(), now.date()]
    lag = 0
    if latest.date() < now.date():
        for d in pd.date_range(
            latest.date() + timedelta(days=1), now.date(), freq="D"
        ):
            if market_calendar.is_trading_day(d.date()):
                lag += 1
    status["lag_trading_days"] = lag

    if lag > threshold:
        status["stale"] = True
        status["detail"] = (
            f"Most recent daily VOO candle ({status['latest_daily']}) is "
            f"{lag} trading days old (threshold {threshold}) — "
            "the ingestion pipeline may have stopped writing new candles."
        )
        logger.error(
            "candle_feed_stale latest_daily=%s lag_trading_days=%d threshold=%d — %s",
            status["latest_daily"], lag, threshold, status["detail"],
        )

    return status


async def check_5min_staleness(
    db_session: AsyncSession,
    now: Optional[datetime] = None,
) -> dict:
    """
    Detect a quietly-stalled VOO 5-minute feed during market hours.

    Only meaningful while the regular market session is open (per
    ingestion.market_calendar): when open, the latest stored 5-min bar must
    be no older than settings.FIVEMIN_STALENESS_MAX_AGE_MINUTES minutes.
    The first max-age minutes after the open are grace time (yesterday's
    last bar is legitimately the latest one right at the bell).

    Logs a WARNING (event `fivemin_data_stale`) when stale and returns a
    structured dict for the health endpoint:

        {
            "stale": bool,
            "market_open": bool,
            "latest_5min": Optional[str],        # ISO timestamp (UTC)
            "age_minutes": Optional[float],
            "max_age_minutes": int,
            "detail": Optional[str],             # set when stale
        }

    `now` is injectable for tests; defaults to current UTC (naive, matching
    the DB's UTC-naive timestamps).
    """
    if now is None:
        now = datetime.utcnow()
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)

    max_age = settings.FIVEMIN_STALENESS_MAX_AGE_MINUTES

    result = await db_session.execute(
        select(func.max(VooCandle.timestamp)).where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == "5min",
        )
    )
    latest_5min: Optional[datetime] = result.scalar()

    is_extended, session_type, _ = market_calendar.classify_session(now)
    market_open = session_type == "regular" and not is_extended

    status: dict = {
        "stale": False,
        "market_open": market_open,
        "latest_5min": latest_5min.isoformat() if latest_5min else None,
        "age_minutes": (
            round((now - latest_5min).total_seconds() / 60.0, 1)
            if latest_5min else None
        ),
        "max_age_minutes": max_age,
        "detail": None,
    }

    if not market_open:
        # Outside regular hours a quiet feed is expected — never alarm.
        return status

    # Grace period right after the open: the latest bar may legitimately be
    # yesterday's last bar until the first bars of today arrive.
    now_et = market_calendar.to_eastern(now)
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_since_open = (now_et - open_et).total_seconds() / 60.0
    if minutes_since_open < max_age:
        return status

    if latest_5min is None:
        status["stale"] = True
        status["detail"] = (
            "No VOO 5-min bars stored while the market is open — "
            "the short-trend signal has no intraday data."
        )
        logger.warning(
            "fivemin_data_stale latest_5min=none — %s", status["detail"]
        )
        return status

    if status["age_minutes"] > max_age:
        status["stale"] = True
        status["detail"] = (
            f"Latest VOO 5-min bar ({status['latest_5min']}) is "
            f"{status['age_minutes']:.1f} minutes old during market hours "
            f"(max {max_age}) — the short-trend signal is running on stale "
            f"intraday data."
        )
        logger.warning(
            "fivemin_data_stale latest_5min=%s age_minutes=%.1f max=%d — %s",
            status["latest_5min"], status["age_minutes"], max_age,
            status["detail"],
        )
    return status
