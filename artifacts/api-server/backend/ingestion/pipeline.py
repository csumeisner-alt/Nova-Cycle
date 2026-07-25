"""
NovaCycle Ingestion Pipeline
=============================
Orchestrates data fetching, gap/session tagging, and DB storage.

Schedule:
  - Every 5 min during market hours (09:25 – 20:05 ET Mon–Fri)
  - Daily after close for daily candles

NOTE: "Pipeline currently fetches only VOO. Multi-ticker ingestion will be added later."
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.models import VooCandle, VixCandle
from ingestion import market_calendar
from ingestion.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Manages full and incremental ingestion of VOO and VIX market data."""

    def __init__(self):
        self.fetcher = DataFetcher()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self, db_session: AsyncSession) -> None:
        """
        Smart initialisation:
          - If no VOO daily data in DB → run full historical fetch + store
          - If data exists → run incremental update only
        """
        logger.info("IngestionPipeline.initialize() called")

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
        result = await db_session.execute(
            select(func.max(VixCandle.timestamp)).where(
                VixCandle.ticker == settings.VIX_TICKER
            )
        )
        last_vix: Optional[datetime] = result.scalar()

        if last_vix:
            vix_df = await self.fetcher.fetch_historical_vix(years=1)
            if not vix_df.empty:
                vix_df = vix_df[vix_df.index > pd.Timestamp(last_vix)]
                if not vix_df.empty:
                    await self.store_vix_candles(vix_df, db_session, timeframe="daily")

        logger.info("Incremental update complete.")

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

        # ── Missing-candle detection + targeted backfill (daily only) ─────────
        # Detect trading days with no candle in the fetched window, then
        # proactively re-fetch just those date ranges. Backfill failures are
        # logged and never abort the regular ingestion run.
        missing_days: list = []
        try:
            if timeframe == "daily" and len(candles) > 1 and not _is_backfill:
                idx = candles.sort_index().index
                have = {ts.date() for ts in idx}
                missing_days = [
                    d.date()
                    for d in pd.date_range(idx[0].date(), idx[-1].date(), freq="D")
                    if market_calendar.is_trading_day(d.date())
                    and d.date() not in have
                ]
                if missing_days:
                    logger.warning(
                        "ingest_missing_candles timeframe=daily count=%d days=%s",
                        len(missing_days),
                        ",".join(d.isoformat() for d in missing_days[:20]),
                    )
        except Exception as exc:
            logger.error("ingest_missing_check_failed error=%s", exc)
            missing_days = []

        # Pre-load existing timestamps to avoid duplicate queries
        result = await db_session.execute(
            select(VooCandle.timestamp).where(
                VooCandle.ticker == ticker,
                VooCandle.timeframe == timeframe,
            )
        )
        existing_timestamps = set(row[0] for row in result.fetchall())

        # Iterate sorted by time (oldest first for gap calc)
        prev_close: Optional[float] = None

        for ts, row in candles.sort_index().iterrows():
            # Normalise timestamp
            ts_naive = ts.to_pydatetime()
            if ts_naive.tzinfo is not None:
                ts_naive = ts_naive.replace(tzinfo=None)

            if ts_naive in existing_timestamps:
                skipped += 1
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
                open=float(row.get("open", 0.0)),
                high=float(row.get("high", 0.0)),
                low=float(row.get("low", 0.0)),
                close=float(row.get("close", 0.0)),
                volume=float(row.get("volume", 0.0)),
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
            "VOO %s candles: inserted=%d, skipped=%d (duplicates)",
            timeframe, inserted, skipped,
        )

        # ── Targeted backfill of missing trading days ─────────────────────────
        if missing_days:
            try:
                await self._backfill_missing_days(missing_days, db_session)
            except Exception as exc:
                logger.error("ingest_backfill_failed error=%s", exc)

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

    async def _backfill_missing_days(self, missing_days: list, db_session: AsyncSession) -> None:
        """
        Re-fetch and store daily candles for the given missing trading days.

        Failures are logged per range and never propagate to the caller's
        regular ingestion flow.
        """
        from datetime import datetime as _dt, time as _time

        ranges = self._group_contiguous_days(missing_days)
        logger.info(
            "ingest_backfill_start days=%d ranges=%d",
            len(missing_days), len(ranges),
        )

        filled = 0
        for start_d, end_d in ranges:
            try:
                start = _dt.combine(start_d, _time.min)
                end = _dt.combine(end_d, _time.min)
                df = await self.fetcher.fetch_daily_range(start, end)
                if df.empty:
                    logger.warning(
                        "ingest_backfill_empty range=%s→%s",
                        start_d.isoformat(), end_d.isoformat(),
                    )
                    continue
                await self.store_voo_candles(
                    df, db_session, timeframe="daily", _is_backfill=True
                )
                filled += 1
            except Exception as exc:
                logger.error(
                    "ingest_backfill_range_failed range=%s→%s error=%s",
                    start_d.isoformat(), end_d.isoformat(), exc,
                )

        logger.info(
            "ingest_backfill_complete ranges_ok=%d ranges_total=%d",
            filled, len(ranges),
        )

    async def store_vix_candles(
        self,
        candles: pd.DataFrame,
        db_session: AsyncSession,
        timeframe: str,
    ) -> None:
        """
        Persist VIX candles to DB, skipping duplicates.
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

        for ts, row in candles.sort_index().iterrows():
            ts_naive = ts.to_pydatetime()
            if ts_naive.tzinfo is not None:
                ts_naive = ts_naive.replace(tzinfo=None)

            if ts_naive in existing_timestamps:
                skipped += 1
                continue

            candle = VixCandle(
                ticker=ticker,
                timestamp=ts_naive,
                open=float(row.get("open", 0.0)),
                high=float(row.get("high", 0.0)),
                low=float(row.get("low", 0.0)),
                close=float(row.get("close", 0.0)),
                volume=float(row.get("volume", 0.0)),
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
