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
                gap_info = await self.fetcher.detect_gap(prev_close, open_price)
                gap_percent = gap_info["gap_percent"]
                gap_type = gap_info["gap_type"]

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
