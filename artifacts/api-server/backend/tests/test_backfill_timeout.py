"""
Regression tests: a slow (timed-out) yfinance backfill fetch must be logged
and must NOT delay or abort the main ingestion run.

Covers _backfill_missing_days (VOO daily + 5-min), _backfill_missing_vix_days,
and _backfill_missing_context_days for all three fetch paths.
"""

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from config import settings
from database.models import (
    Base, VooCandle, VixCandle,
    VixShortCandle, VixLongCandle, RatesCandle,
    CreditHyCandle, CreditIgCandle, BreadthCandle,
)
from ingestion.pipeline import IngestionPipeline, BACKFILL_FETCH_TIMEOUT_SECS


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///file::memory:?cache=shared",
                                  connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


# Week: Mon 2026-07-06 .. Fri 2026-07-10
WEEK = [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8),
        date(2026, 7, 9), date(2026, 7, 10)]


def make_daily_df(dates: list[date]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates])
    n = len(dates)
    df = pd.DataFrame(
        {
            "open":   [100.0 + i for i in range(n)],
            "high":   [101.0 + i for i in range(n)],
            "low":    [99.0  + i for i in range(n)],
            "close":  [100.5 + i for i in range(n)],
            "volume": [1_000.0] * n,
        },
        index=idx,
    )
    df["is_extended_hours"] = False
    df["session_type"] = "regular"
    return df


def make_vix_df(dates: list[date]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates])
    n = len(dates)
    return pd.DataFrame(
        {
            "open":   [15.0 + i for i in range(n)],
            "high":   [16.0 + i for i in range(n)],
            "low":    [14.0 + i for i in range(n)],
            "close":  [15.5 + i for i in range(n)],
            "volume": [1_000.0] * n,
        },
        index=idx,
    )


async def _slow_fetch(*_args, **_kwargs) -> pd.DataFrame:
    """Simulates a yfinance call that never returns (blocks indefinitely)."""
    await asyncio.sleep(9999)
    return pd.DataFrame()


async def count_rows(db_session, model, ticker, timeframe="daily") -> int:
    result = await db_session.execute(
        select(func.count(model.id)).where(
            model.ticker == ticker,
            model.timeframe == timeframe,
        )
    )
    return result.scalar() or 0


# ─────────────────────────────────────────────────────────────────────────────
# VOO daily backfill timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestVooBackfillTimeout:
    @pytest.mark.asyncio
    async def test_timeout_logged_and_main_rows_preserved(self, db_session, caplog):
        """A hung fetch_daily_range times out: error is logged, main rows kept."""
        pipeline = IngestionPipeline()
        # Mon + Thu + Fri present; Tue + Wed missing → one backfill range
        present = [WEEK[0], WEEK[3], WEEK[4]]
        df = make_daily_df(present)

        with patch.object(pipeline.fetcher, "fetch_daily_range",
                          side_effect=_slow_fetch):
            with patch("ingestion.pipeline.BACKFILL_FETCH_TIMEOUT_SECS", 0.05):
                with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                    await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        # Main rows must be in the DB
        n = await count_rows(db_session, VooCandle, settings.TICKER, "daily")
        assert n == 3

        # Timeout error must be logged
        assert any("ingest_backfill_range_timeout" in r.message for r in caplog.records), \
            "Expected 'ingest_backfill_range_timeout' in log"

    @pytest.mark.asyncio
    async def test_second_range_fetched_after_first_times_out(self, db_session, caplog):
        """When the first range times out the second range is still attempted."""
        pipeline = IngestionPipeline()
        # Missing 7/7 and 7/13 → two separate ranges
        present = [date(2026, 7, 6), date(2026, 7, 8), date(2026, 7, 9),
                   date(2026, 7, 10), date(2026, 7, 14)]
        df = make_daily_df(present)

        good_df = make_daily_df([date(2026, 7, 13)])
        call_count = 0

        async def fetch_side_effect(start, end):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(9999)   # first range hangs
            return good_df

        with patch.object(pipeline.fetcher, "fetch_daily_range",
                          side_effect=fetch_side_effect):
            with patch("ingestion.pipeline.BACKFILL_FETCH_TIMEOUT_SECS", 0.05):
                with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                    await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        # 5 present + 1 backfilled (7/13)
        n = await count_rows(db_session, VooCandle, settings.TICKER, "daily")
        assert n == 6
        assert call_count == 2
        assert any("ingest_backfill_range_timeout" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# VOO 5-min backfill timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestFiveMinBackfillTimeout:
    @pytest.mark.asyncio
    async def test_5min_timeout_logged_and_main_rows_preserved(self, db_session, caplog):
        """A hung fetch_5min_range times out: error is logged, main rows kept."""
        from ingestion import market_calendar
        from datetime import timedelta, timezone

        pipeline = IngestionPipeline()

        # Build a small 5-min frame for two recent trading days; one in between missing
        today = datetime.now(timezone.utc).date()
        days: list[date] = []
        d = today - timedelta(days=3)
        while len(days) < 3:
            if market_calendar.is_trading_day(d):
                days.append(d)
            d -= timedelta(days=1)
        days.reverse()
        d0, d1, d2 = days

        def make_5min(dates):
            from datetime import timedelta as _td
            stamps = []
            for dd in dates:
                base = datetime(dd.year, dd.month, dd.day, 14, 30)
                stamps.extend(base + _td(minutes=5 * i) for i in range(3))
            n = len(stamps)
            df = pd.DataFrame(
                {"open": [100.0]*n, "high": [100.5]*n,
                 "low": [99.5]*n, "close": [100.2]*n, "volume": [500.0]*n},
                index=pd.DatetimeIndex(stamps),
            )
            df["is_extended_hours"] = False
            df["session_type"] = "regular"
            return df

        df = make_5min([d0, d2])  # d1 missing → backfill triggered

        with patch.object(pipeline.fetcher, "fetch_5min_range",
                          side_effect=_slow_fetch):
            with patch("ingestion.pipeline.BACKFILL_FETCH_TIMEOUT_SECS", 0.05):
                with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                    await pipeline.store_voo_candles(df, db_session, timeframe="5min")

        n = await count_rows(db_session, VooCandle, settings.TICKER, "5min")
        assert n == 6  # 2 days × 3 bars; no extra from backfill
        assert any("ingest_backfill_range_timeout" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# VIX backfill timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestVixBackfillTimeout:
    @pytest.mark.asyncio
    async def test_vix_timeout_logged_and_main_rows_preserved(self, db_session, caplog):
        """A hung fetch_vix_daily_range times out: error logged, main rows kept."""
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[3], WEEK[4]]
        df = make_vix_df(present)

        with patch.object(pipeline.fetcher, "fetch_vix_daily_range",
                          side_effect=_slow_fetch):
            with patch("ingestion.pipeline.BACKFILL_FETCH_TIMEOUT_SECS", 0.05):
                with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                    await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        n = await count_rows(db_session, VixCandle, settings.VIX_TICKER, "daily")
        assert n == 3
        assert any("vix_ingest_backfill_range_timeout" in r.message for r in caplog.records), \
            "Expected 'vix_ingest_backfill_range_timeout' in log"

    @pytest.mark.asyncio
    async def test_vix_second_range_fetched_after_first_times_out(self, db_session, caplog):
        """When first VIX range times out the second is still attempted."""
        pipeline = IngestionPipeline()
        present = [date(2026, 7, 6), date(2026, 7, 8), date(2026, 7, 9),
                   date(2026, 7, 10), date(2026, 7, 14)]
        df = make_vix_df(present)

        good_df = make_vix_df([date(2026, 7, 13)])
        call_count = 0

        async def fetch_side_effect(start, end):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(9999)
            return good_df

        with patch.object(pipeline.fetcher, "fetch_vix_daily_range",
                          side_effect=fetch_side_effect):
            with patch("ingestion.pipeline.BACKFILL_FETCH_TIMEOUT_SECS", 0.05):
                with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                    await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        n = await count_rows(db_session, VixCandle, settings.VIX_TICKER, "daily")
        assert n == 6
        assert call_count == 2
        assert any("vix_ingest_backfill_range_timeout" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Context ticker backfill timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBackfillTimeout:
    @pytest.mark.asyncio
    async def test_context_timeout_logged_and_main_rows_preserved(self, db_session, caplog):
        """A hung fetch_context_ticker_range times out: error logged, main rows kept."""
        pipeline = IngestionPipeline()

        # Build a small VIX9D frame with a gap
        ticker = settings.VIX_SHORT_TICKER
        idx_present = pd.DatetimeIndex([
            pd.Timestamp(datetime(d.year, d.month, d.day))
            for d in [WEEK[0], WEEK[3], WEEK[4]]
        ])
        n = len(idx_present)
        df = pd.DataFrame(
            {"open": [20.0]*n, "high": [21.0]*n, "low": [19.0]*n,
             "close": [20.5]*n, "volume": [0.0]*n},
            index=idx_present,
        )

        with patch.object(pipeline.fetcher, "fetch_context_ticker_range",
                          side_effect=_slow_fetch):
            with patch("ingestion.pipeline.BACKFILL_FETCH_TIMEOUT_SECS", 0.05):
                with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                    await pipeline.store_context_candles(
                        df, db_session,
                        model=VixShortCandle,
                        ticker=ticker,
                        label="VIX9D",
                        is_index=True,
                    )

        n_rows = await count_rows(db_session, VixShortCandle, ticker, "daily")
        assert n_rows == 3
        assert any("context_ingest_backfill_range_timeout" in r.message for r in caplog.records), \
            "Expected 'context_ingest_backfill_range_timeout' in log"


# ─────────────────────────────────────────────────────────────────────────────
# Constant sanity check
# ─────────────────────────────────────────────────────────────────────────────

def test_backfill_timeout_constant_is_positive():
    assert BACKFILL_FETCH_TIMEOUT_SECS > 0
    assert BACKFILL_FETCH_TIMEOUT_SECS <= 120, "Timeout should be ≤ 120 s to avoid blocking the scheduler"
