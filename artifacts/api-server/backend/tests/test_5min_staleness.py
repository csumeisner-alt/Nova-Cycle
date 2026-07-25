"""Tests for the VOO 5-min feed staleness check (quiet intraday stall detection)."""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VooCandle
from ingestion import market_calendar
from ingestion.pipeline import check_5min_staleness


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _bar(ts):
    return VooCandle(
        ticker=settings.TICKER, timestamp=ts, open=1, high=1, low=1, close=1,
        volume=1, timeframe="5min", is_extended_hours=False,
        session_type="regular", gap_percent=0.0, gap_type="none",
    )


def _midday_trading_utc() -> datetime:
    """A UTC-naive timestamp falling mid regular session on a trading day."""
    d = datetime.utcnow().date()
    while True:
        if market_calendar.is_trading_day(d) and not market_calendar.is_half_day(d):
            # 12:00 ET → UTC
            et_noon = datetime(d.year, d.month, d.day, 12, 0, tzinfo=market_calendar.EASTERN)
            return et_noon.astimezone(market_calendar.timezone.utc).replace(tzinfo=None)
        d -= timedelta(days=1)


NOW = _midday_trading_utc()
MAX_AGE = settings.FIVEMIN_STALENESS_MAX_AGE_MINUTES


@pytest.mark.asyncio
async def test_fresh_bar_not_stale(db_session):
    db_session.add(_bar(NOW - timedelta(minutes=5)))
    await db_session.flush()
    status = await check_5min_staleness(db_session, now=NOW)
    assert status["market_open"] is True
    assert status["stale"] is False
    assert status["age_minutes"] == 5.0


@pytest.mark.asyncio
async def test_old_bar_during_market_hours_is_stale(db_session):
    db_session.add(_bar(NOW - timedelta(minutes=MAX_AGE + 10)))
    await db_session.flush()
    status = await check_5min_staleness(db_session, now=NOW)
    assert status["stale"] is True
    assert status["age_minutes"] > MAX_AGE
    assert "stale" in status["detail"]


@pytest.mark.asyncio
async def test_no_bars_during_market_hours_is_stale(db_session):
    status = await check_5min_staleness(db_session, now=NOW)
    assert status["stale"] is True
    assert "No VOO 5-min bars" in status["detail"]


@pytest.mark.asyncio
async def test_old_bar_outside_market_hours_not_stale(db_session):
    db_session.add(_bar(NOW - timedelta(hours=20)))
    await db_session.flush()
    # 3:00 ET (before pre-market) on the same day → market closed
    d = market_calendar.to_eastern(NOW).date()
    closed_et = datetime(d.year, d.month, d.day, 3, 0, tzinfo=market_calendar.EASTERN)
    closed = closed_et.astimezone(market_calendar.timezone.utc).replace(tzinfo=None)
    status = await check_5min_staleness(db_session, now=closed)
    assert status["market_open"] is False
    assert status["stale"] is False


@pytest.mark.asyncio
async def test_grace_period_right_after_open(db_session):
    # A few minutes after the 9:30 ET open, yesterday's last bar is fine.
    d = market_calendar.to_eastern(NOW).date()
    just_open_et = datetime(d.year, d.month, d.day, 9, 35, tzinfo=market_calendar.EASTERN)
    just_open = just_open_et.astimezone(market_calendar.timezone.utc).replace(tzinfo=None)
    db_session.add(_bar(just_open - timedelta(hours=17)))
    await db_session.flush()
    status = await check_5min_staleness(db_session, now=just_open)
    assert status["market_open"] is True
    assert status["stale"] is False


@pytest.mark.asyncio
async def test_exactly_at_threshold_not_stale(db_session):
    db_session.add(_bar(NOW - timedelta(minutes=MAX_AGE)))
    await db_session.flush()
    status = await check_5min_staleness(db_session, now=NOW)
    assert status["stale"] is False
    assert status["age_minutes"] == float(MAX_AGE)
