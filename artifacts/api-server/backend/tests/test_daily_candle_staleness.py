"""Tests for the daily VOO candle feed staleness check."""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VooCandle
from ingestion import market_calendar
from ingestion.pipeline import check_daily_candle_staleness


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _voo_daily(ts: datetime) -> VooCandle:
    return VooCandle(
        ticker=settings.TICKER,
        timestamp=ts,
        open=400.0,
        high=402.0,
        low=399.0,
        close=401.0,
        volume=1_000_000,
        timeframe="daily",
        is_extended_hours=False,
        session_type="regular",
        gap_percent=0.0,
        gap_type="none",
    )


def _recent_trading_days(n: int) -> list:
    """Return the n most recent trading days as naive UTC datetimes, newest last."""
    days, d = [], datetime.utcnow().date()
    while len(days) < n:
        if market_calendar.is_trading_day(d):
            days.append(datetime(d.year, d.month, d.day))
        d -= timedelta(days=1)
    return list(reversed(days))


# ── No data ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_candles_not_stale(db_session):
    """An empty table is not flagged as stale (fresh deployment, not a stopped feed)."""
    status = await check_daily_candle_staleness(db_session)
    assert status["stale"] is False
    assert status["latest_daily"] is None
    assert status["lag_trading_days"] is None


# ── Fresh feed ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fresh_feed_not_stale(db_session):
    """A candle written on the most recent trading day is not stale."""
    days = _recent_trading_days(3)
    for ts in days:
        db_session.add(_voo_daily(ts))
    await db_session.flush()

    now = days[-1] + timedelta(hours=16)  # same day, after market close
    status = await check_daily_candle_staleness(db_session, now=now)
    assert status["stale"] is False
    assert status["lag_trading_days"] is not None
    assert status["lag_trading_days"] <= settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS


@pytest.mark.asyncio
async def test_one_trading_day_lag_not_stale(db_session):
    """A lag of exactly 1 trading day is within the default threshold of 3."""
    days = _recent_trading_days(5)
    for ts in days[:-1]:          # store all but the most recent
        db_session.add(_voo_daily(ts))
    await db_session.flush()

    # Simulate 'now' = the latest real trading day + 1 calendar day
    now = days[-1] + timedelta(days=1)
    status = await check_daily_candle_staleness(db_session, now=now)
    assert status["stale"] is False
    assert status["lag_trading_days"] is not None


# ── Stale feed ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feed_stale_when_lag_exceeds_threshold(db_session):
    """When the lag exceeds DAILY_CANDLE_STALE_THRESHOLD_DAYS the feed is stale."""
    threshold = settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS
    days = _recent_trading_days(threshold + 5)
    # Only store the oldest candle (simulate the feed stopping)
    db_session.add(_voo_daily(days[0]))
    await db_session.flush()

    # Use today as 'now' so the lag is naturally large
    now = datetime.utcnow()
    status = await check_daily_candle_staleness(db_session, now=now)
    assert status["stale"] is True
    assert status["lag_trading_days"] is not None
    assert status["lag_trading_days"] > threshold
    assert status["detail"] is not None
    assert "ingestion pipeline" in status["detail"]


@pytest.mark.asyncio
async def test_stale_detail_mentions_lag(db_session):
    """The detail string contains the lag and the latest date."""
    threshold = settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS
    days = _recent_trading_days(threshold + 6)
    db_session.add(_voo_daily(days[0]))
    await db_session.flush()

    status = await check_daily_candle_staleness(db_session, now=datetime.utcnow())
    assert status["stale"] is True
    assert status["latest_daily"] in status["detail"]
    assert str(status["lag_trading_days"]) in status["detail"]


# ── Threshold boundary ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exactly_at_threshold_not_stale(db_session):
    """A lag of exactly the threshold is not stale (> not >=)."""
    threshold = settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS
    # Need threshold+1 trading days so the oldest is exactly `threshold`
    # trading days before the newest.
    days = _recent_trading_days(threshold + 1)
    db_session.add(_voo_daily(days[0]))
    await db_session.flush()

    # 'now' = days[-1] so lag == threshold exactly
    now = days[-1] + timedelta(hours=12)
    status = await check_daily_candle_staleness(db_session, now=now)
    assert status["lag_trading_days"] == threshold
    assert status["stale"] is False


@pytest.mark.asyncio
async def test_one_beyond_threshold_is_stale(db_session):
    """A lag of threshold+1 trading days triggers the stale flag."""
    threshold = settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS
    days = _recent_trading_days(threshold + 2)
    db_session.add(_voo_daily(days[0]))
    await db_session.flush()

    now = days[-1] + timedelta(hours=12)
    status = await check_daily_candle_staleness(db_session, now=now)
    assert status["lag_trading_days"] == threshold + 1
    assert status["stale"] is True


# ── Extended-hours candles excluded ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_extended_hours_candles_ignored(db_session):
    """Extended-hours rows must not be used as the latest daily candle."""
    threshold = settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS
    days = _recent_trading_days(threshold + 5)

    # Store only an extended-hours candle at the most recent date
    ext_candle = VooCandle(
        ticker=settings.TICKER,
        timestamp=days[-1],
        open=400.0, high=402.0, low=399.0, close=401.0,
        volume=100_000,
        timeframe="daily",
        is_extended_hours=True,
        session_type="pre_market",
        gap_percent=0.0,
        gap_type="none",
    )
    db_session.add(ext_candle)
    # The only regular-hours candle is the oldest
    db_session.add(_voo_daily(days[0]))
    await db_session.flush()

    now = datetime.utcnow()
    status = await check_daily_candle_staleness(db_session, now=now)
    # Should be stale because the extended candle is filtered out
    assert status["stale"] is True
    assert status["lag_trading_days"] is not None
    assert status["lag_trading_days"] > threshold


# ── Structured dict contract ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_response_keys_always_present(db_session):
    """The returned dict always has all required keys regardless of state."""
    required_keys = {
        "stale", "latest_daily", "lag_trading_days",
        "threshold_trading_days", "detail",
    }
    # Empty DB
    status = await check_daily_candle_staleness(db_session)
    assert required_keys <= set(status.keys())

    # With candles
    days = _recent_trading_days(2)
    for ts in days:
        db_session.add(_voo_daily(ts))
    await db_session.flush()
    status = await check_daily_candle_staleness(db_session)
    assert required_keys <= set(status.keys())


@pytest.mark.asyncio
async def test_threshold_trading_days_reflects_config(db_session):
    """threshold_trading_days in the response matches the config setting."""
    status = await check_daily_candle_staleness(db_session)
    assert status["threshold_trading_days"] == settings.DAILY_CANDLE_STALE_THRESHOLD_DAYS
