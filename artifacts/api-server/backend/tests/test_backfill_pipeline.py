"""
Regression tests for the automatic backfill of missing market history
(downtime repair): missing-day detection → contiguous range grouping →
targeted fetch → store.

Covers:
  - _group_contiguous_days edge cases (single day, weekend spans, long gaps)
  - Pipeline: injected missing days trigger fetch_daily_range with the right
    ranges and backfilled rows are stored exactly once (no duplicates)
  - Failure injection: backfill fetch errors are logged and the main
    ingestion run still completes and persists its own rows
"""

from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VooCandle
from ingestion.pipeline import IngestionPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def make_daily_df(dates: list[date]) -> pd.DataFrame:
    """Build a daily-candle frame matching fetcher.fetch_daily_range output."""
    idx = pd.DatetimeIndex([pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates])
    n = len(dates)
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000.0] * n,
        },
        index=idx,
    )
    df["is_extended_hours"] = False
    df["session_type"] = "regular"
    return df


async def count_daily(db_session) -> int:
    result = await db_session.execute(
        select(func.count(VooCandle.id)).where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == "daily",
        )
    )
    return result.scalar() or 0


# ─────────────────────────────────────────────────────────────────────────────
# _group_contiguous_days
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupContiguousDays:
    group = staticmethod(IngestionPipeline._group_contiguous_days)

    def test_single_day(self):
        d = date(2026, 7, 6)
        assert self.group([d]) == [(d, d)]

    def test_consecutive_weekdays_one_range(self):
        days = [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)]
        assert self.group(days) == [(days[0], days[-1])]

    def test_weekend_span_stays_one_range(self):
        # Fri 2026-07-10 → Mon 2026-07-13 is a 3-day calendar gap (≤ 4)
        days = [date(2026, 7, 9), date(2026, 7, 10), date(2026, 7, 13)]
        assert self.group(days) == [(date(2026, 7, 9), date(2026, 7, 13))]

    def test_long_weekend_holiday_span_stays_one_range(self):
        # Thu → Mon (4 calendar days apart, e.g. Fri holiday + weekend)
        days = [date(2026, 11, 26), date(2026, 11, 30)]
        assert self.group(days) == [(date(2026, 11, 26), date(2026, 11, 30))]

    def test_long_gap_splits_ranges(self):
        days = [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 20), date(2026, 7, 21)]
        assert self.group(days) == [
            (date(2026, 7, 6), date(2026, 7, 7)),
            (date(2026, 7, 20), date(2026, 7, 21)),
        ]

    def test_five_day_gap_splits(self):
        days = [date(2026, 7, 6), date(2026, 7, 11)]
        assert self.group(days) == [
            (date(2026, 7, 6), date(2026, 7, 6)),
            (date(2026, 7, 11), date(2026, 7, 11)),
        ]

    def test_unsorted_input_is_sorted(self):
        days = [date(2026, 7, 8), date(2026, 7, 6), date(2026, 7, 7)]
        assert self.group(days) == [(date(2026, 7, 6), date(2026, 7, 8))]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline: missing days → targeted fetch → stored once
# ─────────────────────────────────────────────────────────────────────────────

# Window: Mon 2026-07-06 .. Fri 2026-07-10, all trading days.
WEEK = [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8),
        date(2026, 7, 9), date(2026, 7, 10)]


class TestBackfillPipeline:
    @pytest.mark.asyncio
    async def test_missing_days_fetched_and_stored_once(self, db_session):
        pipeline = IngestionPipeline()
        # Frame is missing Tue 7th and Wed 8th (contiguous → one range)
        present = [WEEK[0], WEEK[3], WEEK[4]]
        missing = [WEEK[1], WEEK[2]]
        df = make_daily_df(present)

        fetch_mock = AsyncMock(return_value=make_daily_df(missing))
        with patch.object(pipeline.fetcher, "fetch_daily_range", fetch_mock):
            await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        fetch_mock.assert_awaited_once()
        start_arg, end_arg = fetch_mock.await_args.args
        assert start_arg.date() == missing[0]
        assert end_arg.date() == missing[-1]

        assert await count_daily(db_session) == 5
        # Backfilled days present exactly once
        for d in missing:
            result = await db_session.execute(
                select(func.count(VooCandle.id)).where(
                    VooCandle.timestamp == datetime(d.year, d.month, d.day),
                    VooCandle.timeframe == "daily",
                )
            )
            assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_split_gaps_produce_multiple_fetch_ranges(self, db_session):
        pipeline = IngestionPipeline()
        # Missing Mon 7/6 and Mon 7/13 (7 days apart → two ranges)
        present = [date(2026, 7, 7), date(2026, 7, 8), date(2026, 7, 9),
                   date(2026, 7, 10), date(2026, 7, 14)]
        # Note: window is 7/7..7/14, so missing = 7/13 only... include 7/6 by
        # putting it in the window: add 7/6? Instead use window 7/6..7/14.
        present = [date(2026, 7, 6), date(2026, 7, 8), date(2026, 7, 9),
                   date(2026, 7, 10), date(2026, 7, 14)]
        # Missing trading days inside window: 7/7 and 7/13 (6 days apart → split)
        df = make_daily_df(present)

        fetch_mock = AsyncMock(side_effect=[
            make_daily_df([date(2026, 7, 7)]),
            make_daily_df([date(2026, 7, 13)]),
        ])
        with patch.object(pipeline.fetcher, "fetch_daily_range", fetch_mock):
            await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        assert fetch_mock.await_count == 2
        ranges = [(c.args[0].date(), c.args[1].date()) for c in fetch_mock.await_args_list]
        assert ranges == [
            (date(2026, 7, 7), date(2026, 7, 7)),
            (date(2026, 7, 13), date(2026, 7, 13)),
        ]
        assert await count_daily(db_session) == 7

    @pytest.mark.asyncio
    async def test_no_missing_days_no_fetch(self, db_session):
        pipeline = IngestionPipeline()
        df = make_daily_df(WEEK)
        fetch_mock = AsyncMock()
        with patch.object(pipeline.fetcher, "fetch_daily_range", fetch_mock):
            await pipeline.store_voo_candles(df, db_session, timeframe="daily")
        fetch_mock.assert_not_awaited()
        assert await count_daily(db_session) == 5

    @pytest.mark.asyncio
    async def test_backfill_overlapping_existing_rows_not_duplicated(self, db_session):
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[2], WEEK[4]]   # missing Tue + Thu
        df = make_daily_df(present)

        # Backfill fetch returns the whole week (overlaps existing rows)
        fetch_mock = AsyncMock(return_value=make_daily_df(WEEK))
        with patch.object(pipeline.fetcher, "fetch_daily_range", fetch_mock):
            await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        assert await count_daily(db_session) == 5  # no duplicates

    @pytest.mark.asyncio
    async def test_backfill_does_not_recurse(self, db_session):
        """A backfill frame that itself has holes must not trigger another
        round of backfill (guarded by _is_backfill)."""
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[4]]   # missing Tue/Wed/Thu
        df = make_daily_df(present)

        # Backfill returns only Tue — still leaves Wed/Thu missing
        fetch_mock = AsyncMock(return_value=make_daily_df([WEEK[1]]))
        with patch.object(pipeline.fetcher, "fetch_daily_range", fetch_mock):
            await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        fetch_mock.assert_awaited_once()
        assert await count_daily(db_session) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Failure injection
# ─────────────────────────────────────────────────────────────────────────────

class TestBackfillFailures:
    @pytest.mark.asyncio
    async def test_fetch_exception_never_aborts_main_run(self, db_session, caplog):
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[3], WEEK[4]]   # missing Tue+Wed
        df = make_daily_df(present)

        fetch_mock = AsyncMock(side_effect=RuntimeError("yfinance down"))
        with patch.object(pipeline.fetcher, "fetch_daily_range", fetch_mock):
            with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        # Main run's rows are stored despite the backfill failure
        assert await count_daily(db_session) == 3
        assert any("ingest_backfill_range_failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_one_range_fails_other_still_backfilled(self, db_session, caplog):
        pipeline = IngestionPipeline()
        present = [date(2026, 7, 6), date(2026, 7, 8), date(2026, 7, 9),
                   date(2026, 7, 10), date(2026, 7, 14)]
        # Missing: 7/7 and 7/13 → two ranges; first fails, second succeeds
        df = make_daily_df(present)

        fetch_mock = AsyncMock(side_effect=[
            RuntimeError("boom"),
            make_daily_df([date(2026, 7, 13)]),
        ])
        with patch.object(pipeline.fetcher, "fetch_daily_range", fetch_mock):
            with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        assert fetch_mock.await_count == 2
        assert await count_daily(db_session) == 6  # 5 originals + 7/13
        assert any("ingest_backfill_range_failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_backfill_frame_logged_and_run_completes(self, db_session, caplog):
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[3], WEEK[4]]
        df = make_daily_df(present)

        fetch_mock = AsyncMock(return_value=pd.DataFrame())
        with patch.object(pipeline.fetcher, "fetch_daily_range", fetch_mock):
            with caplog.at_level("WARNING", logger="ingestion.pipeline"):
                await pipeline.store_voo_candles(df, db_session, timeframe="daily")

        assert await count_daily(db_session) == 3
        assert any("ingest_backfill_empty" in r.message for r in caplog.records)
