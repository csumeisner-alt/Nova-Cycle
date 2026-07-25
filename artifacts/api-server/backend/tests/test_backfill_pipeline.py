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

from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VooCandle
from ingestion import market_calendar
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


# ─────────────────────────────────────────────────────────────────────────────
# 5-minute (intraday) backfill path
# ─────────────────────────────────────────────────────────────────────────────
#
# The 5-min branch differs from the daily one in three ways, each covered
# below:
#   1. DB sessions are merged into the "have" set → downtime gaps (days in
#      neither the DB nor the fetched frame) are detected.
#   2. The window is clamped to yfinance's ~60-day 5-min limit (58-day floor).
#   3. fetch_5min_range (not fetch_daily_range) is used for the backfill.
#
# All dates here are computed relative to "today" so the tests stay valid
# regardless of when they run (the pipeline's fetch floor uses now()).

FETCH_FLOOR_DAYS = 58


def trading_days_back(count: int, end_days_ago: int = 3) -> list[date]:
    """Return `count` consecutive trading days ending ~end_days_ago days ago,
    oldest first."""
    days: list[date] = []
    d = datetime.now(timezone.utc).date() - timedelta(days=end_days_ago)
    while len(days) < count:
        if market_calendar.is_trading_day(d):
            days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


def trading_days_between(start: date, end: date) -> list[date]:
    """All trading days in [start, end], oldest first."""
    out = []
    d = start
    while d <= end:
        if market_calendar.is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def make_5min_df(dates: list[date], bars_per_day: int = 3) -> pd.DataFrame:
    """Build a 5-min candle frame (regular-session bars, UTC-naive) matching
    fetcher.fetch_5min_range output."""
    stamps = []
    for d in dates:
        base = datetime(d.year, d.month, d.day, 14, 30)  # ~09:30 ET in UTC
        stamps.extend(base + timedelta(minutes=5 * i) for i in range(bars_per_day))
    n = len(stamps)
    df = pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [100.5 + i * 0.1 for i in range(n)],
            "low": [99.5 + i * 0.1 for i in range(n)],
            "close": [100.2 + i * 0.1 for i in range(n)],
            "volume": [500.0] * n,
        },
        index=pd.DatetimeIndex(stamps),
    )
    df["is_extended_hours"] = False
    df["session_type"] = "regular"
    return df


async def insert_5min_rows(db_session, dates: list[date], bars_per_day: int = 3):
    """Pre-seed the DB with 5-min candles for the given sessions."""
    for d in dates:
        base = datetime(d.year, d.month, d.day, 14, 30)
        for i in range(bars_per_day):
            db_session.add(VooCandle(
                ticker=settings.TICKER,
                timestamp=base + timedelta(minutes=5 * i),
                open=100.0, high=100.5, low=99.5, close=100.2, volume=500.0,
                timeframe="5min",
                is_extended_hours=False,
                session_type="regular",
                gap_percent=0.0,
                gap_type="none",
            ))
    await db_session.flush()


async def count_5min(db_session) -> int:
    result = await db_session.execute(
        select(func.count(VooCandle.id)).where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == "5min",
        )
    )
    return result.scalar() or 0


async def count_5min_on_day(db_session, d: date) -> int:
    start = datetime(d.year, d.month, d.day)
    end = start + timedelta(days=1)
    result = await db_session.execute(
        select(func.count(VooCandle.id)).where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == "5min",
            VooCandle.timestamp >= start,
            VooCandle.timestamp < end,
        )
    )
    return result.scalar() or 0


class TestFiveMinBackfill:
    @pytest.mark.asyncio
    async def test_in_frame_gap_fetched_with_5min_range_and_stored_once(self, db_session):
        """A trading session missing inside the fetched frame triggers
        fetch_5min_range (not fetch_daily_range) with the right range."""
        pipeline = IngestionPipeline()
        d0, d1, d2 = trading_days_back(3)
        df = make_5min_df([d0, d2])                     # d1 missing in-frame

        fetch_5m = AsyncMock(return_value=make_5min_df([d1]))
        fetch_daily = AsyncMock()
        with patch.object(pipeline.fetcher, "fetch_5min_range", fetch_5m), \
             patch.object(pipeline.fetcher, "fetch_daily_range", fetch_daily):
            await pipeline.store_voo_candles(df, db_session, timeframe="5min")

        fetch_daily.assert_not_awaited()
        fetch_5m.assert_awaited_once()
        start_arg, end_arg = fetch_5m.await_args.args
        assert start_arg.date() == d1
        assert end_arg.date() == d1

        assert await count_5min(db_session) == 9        # 3 days × 3 bars
        assert await count_5min_on_day(db_session, d1) == 3

    @pytest.mark.asyncio
    async def test_db_only_gap_detected_after_downtime(self, db_session):
        """Sessions in neither the DB nor the current frame (classic downtime
        hole) are detected because DB sessions extend the window."""
        pipeline = IngestionPipeline()
        d0, d1, d2, d3 = trading_days_back(4)
        await insert_5min_rows(db_session, [d0])        # DB knows only d0
        df = make_5min_df([d2, d3])                     # frame starts at d2 → d1 is a DB-only gap

        fetch_5m = AsyncMock(return_value=make_5min_df([d1]))
        with patch.object(pipeline.fetcher, "fetch_5min_range", fetch_5m):
            await pipeline.store_voo_candles(df, db_session, timeframe="5min")

        fetch_5m.assert_awaited_once()
        start_arg, end_arg = fetch_5m.await_args.args
        assert start_arg.date() == d1
        assert end_arg.date() == d1
        assert await count_5min_on_day(db_session, d1) == 3
        assert await count_5min(db_session) == 12       # d0..d3, 3 bars each

    @pytest.mark.asyncio
    async def test_backfill_overlap_with_db_rows_not_duplicated(self, db_session):
        """Backfill frames overlapping already-stored sessions insert nothing
        twice."""
        pipeline = IngestionPipeline()
        d0, d1, d2 = trading_days_back(3)
        await insert_5min_rows(db_session, [d0])
        df = make_5min_df([d2])                         # d1 missing

        # Backfill returns d0+d1+d2 (overlaps DB row d0 and frame row d2)
        fetch_5m = AsyncMock(return_value=make_5min_df([d0, d1, d2]))
        with patch.object(pipeline.fetcher, "fetch_5min_range", fetch_5m):
            await pipeline.store_voo_candles(df, db_session, timeframe="5min")

        assert await count_5min(db_session) == 9        # each day exactly once
        for d in (d0, d1, d2):
            assert await count_5min_on_day(db_session, d) == 3

    @pytest.mark.asyncio
    async def test_58_day_fetch_floor_respected(self, db_session):
        """Sessions older than the ~60-day yfinance limit are never requested:
        a DB-only gap beyond the floor is left alone, and no requested range
        starts before the floor."""
        pipeline = IngestionPipeline()
        today = datetime.now(timezone.utc).date()
        floor = today - timedelta(days=FETCH_FLOOR_DAYS)

        # One ancient session well beyond the floor (gap right after it is
        # unfetchable and must NOT be requested).
        ancient = next(
            d for d in (floor - timedelta(days=k) for k in range(20, 40))
            if market_calendar.is_trading_day(d)
        )
        await insert_5min_rows(db_session, [ancient], bars_per_day=1)

        # Frame covers every trading day from the floor to a few days ago,
        # except one recent hole → only that hole is fetchable.
        recent = trading_days_between(floor, today - timedelta(days=3))
        hole = recent[len(recent) // 2]
        df = make_5min_df([d for d in recent if d != hole], bars_per_day=1)

        fetch_5m = AsyncMock(return_value=make_5min_df([hole], bars_per_day=1))
        with patch.object(pipeline.fetcher, "fetch_5min_range", fetch_5m):
            await pipeline.store_voo_candles(df, db_session, timeframe="5min")

        # Every requested range lies entirely on/after the floor
        assert fetch_5m.await_count >= 1
        for call in fetch_5m.await_args_list:
            assert call.args[0].date() >= floor
            assert call.args[1].date() >= call.args[0].date()
        # The recent hole was requested; nothing before the floor was
        requested_days = set()
        for call in fetch_5m.await_args_list:
            requested_days |= set(trading_days_between(call.args[0].date(), call.args[1].date()))
        assert hole in requested_days
        assert all(d >= floor for d in requested_days)

    @pytest.mark.asyncio
    async def test_5min_fetch_failure_never_aborts_main_run(self, db_session, caplog):
        pipeline = IngestionPipeline()
        d0, d1, d2 = trading_days_back(3)
        df = make_5min_df([d0, d2])                     # d1 missing

        fetch_5m = AsyncMock(side_effect=RuntimeError("yfinance down"))
        with patch.object(pipeline.fetcher, "fetch_5min_range", fetch_5m):
            with caplog.at_level("ERROR", logger="ingestion.pipeline"):
                await pipeline.store_voo_candles(df, db_session, timeframe="5min")

        assert await count_5min(db_session) == 6        # main run rows kept
        assert any("ingest_backfill_range_failed" in r.message for r in caplog.records)
