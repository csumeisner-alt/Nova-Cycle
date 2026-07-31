"""
Regression tests for automatic backfill of missing VIX daily history
(downtime repair), mirroring tests/test_backfill_pipeline.py for VOO:
missing-day detection → contiguous range grouping → targeted fetch → store.

Covers:
  - Pipeline: missing days in the fetched frame trigger fetch_vix_daily_range
    with the right ranges and backfilled rows are stored exactly once
  - DB-only gaps (downtime holes) inside the frame window are detected
  - Failure injection: backfill fetch errors are logged and the main
    ingestion run still completes and persists its own rows
  - Backfill frames never recurse into another backfill round
"""

from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VixCandle
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


def make_vix_df(dates: list[date]) -> pd.DataFrame:
    """Build a daily VIX frame matching fetcher.fetch_vix_daily_range output."""
    idx = pd.DatetimeIndex([pd.Timestamp(datetime(d.year, d.month, d.day)) for d in dates])
    n = len(dates)
    return pd.DataFrame(
        {
            "open": [15.0 + i for i in range(n)],
            "high": [16.0 + i for i in range(n)],
            "low": [14.0 + i for i in range(n)],
            "close": [15.5 + i for i in range(n)],
            "volume": [1_000.0] * n,
        },
        index=idx,
    )


async def insert_vix_rows(db_session, dates: list[date]):
    for d in dates:
        db_session.add(VixCandle(
            ticker=settings.VIX_TICKER,
            timestamp=datetime(d.year, d.month, d.day),
            open=15.0, high=16.0, low=14.0, close=15.5, volume=1_000.0,
            timeframe="daily",
        ))
    await db_session.flush()


async def count_vix(db_session) -> int:
    result = await db_session.execute(
        select(func.count(VixCandle.id)).where(
            VixCandle.ticker == settings.VIX_TICKER,
            VixCandle.timeframe == "daily",
        )
    )
    return result.scalar() or 0


async def count_vix_on_day(db_session, d: date) -> int:
    result = await db_session.execute(
        select(func.count(VixCandle.id)).where(
            VixCandle.ticker == settings.VIX_TICKER,
            VixCandle.timeframe == "daily",
            VixCandle.timestamp == datetime(d.year, d.month, d.day),
        )
    )
    return result.scalar() or 0


# Window: Mon 2026-07-06 .. Fri 2026-07-10, all trading days.
WEEK = [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8),
        date(2026, 7, 9), date(2026, 7, 10)]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline: missing days → targeted fetch → stored once
# ─────────────────────────────────────────────────────────────────────────────

class TestVixBackfillPipeline:
    @pytest.mark.asyncio
    async def test_missing_days_fetched_and_stored_once(self, db_session):
        pipeline = IngestionPipeline()
        # Frame is missing Tue 7th and Wed 8th (contiguous → one range)
        present = [WEEK[0], WEEK[3], WEEK[4]]
        missing = [WEEK[1], WEEK[2]]
        df = make_vix_df(present)

        fetch_mock = AsyncMock(return_value=make_vix_df(missing))
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        fetch_mock.assert_awaited_once()
        start_arg, end_arg = fetch_mock.await_args.args
        assert start_arg.date() == missing[0]
        assert end_arg.date() == missing[-1]

        assert await count_vix(db_session) == 5
        for d in missing:
            assert await count_vix_on_day(db_session, d) == 1

    @pytest.mark.asyncio
    async def test_split_gaps_produce_multiple_fetch_ranges(self, db_session):
        pipeline = IngestionPipeline()
        # Missing trading days inside window: 7/7 and 7/13 (6 days apart → split)
        present = [date(2026, 7, 6), date(2026, 7, 8), date(2026, 7, 9),
                   date(2026, 7, 10), date(2026, 7, 14)]
        df = make_vix_df(present)

        fetch_mock = AsyncMock(side_effect=[
            make_vix_df([date(2026, 7, 7)]),
            make_vix_df([date(2026, 7, 13)]),
        ])
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        assert fetch_mock.await_count == 2
        ranges = [(c.args[0].date(), c.args[1].date()) for c in fetch_mock.await_args_list]
        assert ranges == [
            (date(2026, 7, 7), date(2026, 7, 7)),
            (date(2026, 7, 13), date(2026, 7, 13)),
        ]
        assert await count_vix(db_session) == 7

    @pytest.mark.asyncio
    async def test_no_missing_days_no_fetch(self, db_session):
        pipeline = IngestionPipeline()
        df = make_vix_df(WEEK)
        fetch_mock = AsyncMock()
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            await pipeline.store_vix_candles(df, db_session, timeframe="daily")
        fetch_mock.assert_not_awaited()
        assert await count_vix(db_session) == 5

    @pytest.mark.asyncio
    async def test_db_only_gap_detected_after_downtime(self, db_session):
        """Days already in the DB extend `have`, so a day present in the DB is
        not treated as missing, while a true hole (in neither DB nor frame)
        inside the window is."""
        pipeline = IngestionPipeline()
        await insert_vix_rows(db_session, [WEEK[1]])   # Tue already in DB
        present = [WEEK[0], WEEK[3], WEEK[4]]          # frame missing Tue+Wed
        df = make_vix_df(present)

        # Only Wed is a true hole
        fetch_mock = AsyncMock(return_value=make_vix_df([WEEK[2]]))
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        fetch_mock.assert_awaited_once()
        start_arg, end_arg = fetch_mock.await_args.args
        assert start_arg.date() == WEEK[2]
        assert end_arg.date() == WEEK[2]
        assert await count_vix(db_session) == 5

    @pytest.mark.asyncio
    async def test_backfill_overlapping_existing_rows_not_duplicated(self, db_session):
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[2], WEEK[4]]   # missing Tue + Thu
        df = make_vix_df(present)

        # Backfill fetch returns the whole week (overlaps existing rows)
        fetch_mock = AsyncMock(return_value=make_vix_df(WEEK))
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        assert await count_vix(db_session) == 5  # no duplicates

    @pytest.mark.asyncio
    async def test_backfill_does_not_recurse(self, db_session):
        """A backfill frame that itself has holes must not trigger another
        round of backfill (guarded by _is_backfill)."""
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[4]]   # missing Tue/Wed/Thu
        df = make_vix_df(present)

        # Backfill returns only Tue — still leaves Wed/Thu missing
        fetch_mock = AsyncMock(return_value=make_vix_df([WEEK[1]]))
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        fetch_mock.assert_awaited_once()
        assert await count_vix(db_session) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Failure injection
# ─────────────────────────────────────────────────────────────────────────────

class TestVixBackfillFailures:
    @pytest.mark.asyncio
    async def test_fetch_exception_never_aborts_main_run(self, db_session, caplog):
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[3], WEEK[4]]   # missing Tue+Wed
        df = make_vix_df(present)

        fetch_mock = AsyncMock(side_effect=RuntimeError("yfinance down"))
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        assert await count_vix(db_session) == 3  # main run rows kept
        assert any("vix_ingest_backfill_range_failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_one_range_fails_other_still_backfilled(self, db_session, caplog):
        pipeline = IngestionPipeline()
        present = [date(2026, 7, 6), date(2026, 7, 8), date(2026, 7, 9),
                   date(2026, 7, 10), date(2026, 7, 14)]
        # Missing: 7/7 and 7/13 → two ranges; first fails, second succeeds
        df = make_vix_df(present)

        fetch_mock = AsyncMock(side_effect=[
            RuntimeError("boom"),
            make_vix_df([date(2026, 7, 13)]),
        ])
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        assert fetch_mock.await_count == 2
        assert await count_vix(db_session) == 6  # 5 originals + 7/13
        assert any("vix_ingest_backfill_range_failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_backfill_frame_logged_and_run_completes(self, db_session, caplog):
        pipeline = IngestionPipeline()
        present = [WEEK[0], WEEK[3], WEEK[4]]
        df = make_vix_df(present)

        fetch_mock = AsyncMock(return_value=pd.DataFrame())
        with patch.object(pipeline.fetcher, "fetch_vix_daily_range", fetch_mock):
            with caplog.at_level("WARNING", logger="ingestion.pipeline"):
                await pipeline.store_vix_candles(df, db_session, timeframe="daily")

        assert await count_vix(db_session) == 3
        assert any("vix_ingest_backfill_empty" in r.message for r in caplog.records)


class TestVixIncrementalRecovery:
    @pytest.mark.asyncio
    async def test_empty_vix_table_is_repopulated_during_incremental_update(
        self, db_session
    ):
        """An empty VIX table must not permanently skip future VIX fetches."""
        pipeline = IngestionPipeline()
        pipeline.fetcher.fetch_incremental_voo = AsyncMock(
            return_value={"daily": pd.DataFrame(), "5min": pd.DataFrame()}
        )
        pipeline.fetcher.fetch_historical_vix = AsyncMock(
            return_value=make_vix_df(WEEK[-2:])
        )
        pipeline.fetcher.fetch_historical_spx = AsyncMock(
            return_value=pd.DataFrame()
        )

        await pipeline.run_incremental_update(db_session)

        pipeline.fetcher.fetch_historical_vix.assert_awaited_once_with(years=1)
        assert await count_vix(db_session) == 2

    @pytest.mark.asyncio
    async def test_vix_fetch_failure_preserves_degraded_data_behavior(
        self, db_session, caplog
    ):
        """A VIX vendor failure is logged and does not abort the update."""
        pipeline = IngestionPipeline()
        pipeline.fetcher.fetch_incremental_voo = AsyncMock(
            return_value={"daily": pd.DataFrame(), "5min": pd.DataFrame()}
        )
        pipeline.fetcher.fetch_historical_vix = AsyncMock(
            side_effect=RuntimeError("VIX vendor unavailable")
        )
        pipeline.fetcher.fetch_historical_spx = AsyncMock(
            return_value=pd.DataFrame()
        )

        with caplog.at_level("ERROR", logger="ingestion.pipeline"):
            await pipeline.run_incremental_update(db_session)

        assert await count_vix(db_session) == 0
        assert any("vix_incremental_fetch_failed" in r.message for r in caplog.records)
