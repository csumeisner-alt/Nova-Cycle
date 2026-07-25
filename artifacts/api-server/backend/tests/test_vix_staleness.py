"""Tests for the VIX staleness check (quiet-data-stop detection)."""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VixCandle, VooCandle
from ingestion import market_calendar
from ingestion.pipeline import check_vix_staleness


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _voo(ts):
    return VooCandle(
        ticker=settings.TICKER, timestamp=ts, open=1, high=1, low=1, close=1,
        volume=1, timeframe="daily", is_extended_hours=False,
        session_type="regular", gap_percent=0.0, gap_type="none",
    )


def _vix(ts):
    return VixCandle(
        ticker=settings.VIX_TICKER, timestamp=ts, open=1, high=1,
        low=1, close=1, volume=1, timeframe="daily",
    )


def _recent_trading_days(n):
    days, d = [], datetime.utcnow().date()
    while len(days) < n:
        if market_calendar.is_trading_day(d):
            days.append(datetime(d.year, d.month, d.day))
        d -= timedelta(days=1)
    return list(reversed(days))


@pytest.mark.asyncio
async def test_no_data_at_all_not_stale(db_session):
    status = await check_vix_staleness(db_session)
    assert status["stale"] is False
    assert status["latest_voo"] is None


@pytest.mark.asyncio
async def test_missing_vix_with_voo_is_stale(db_session):
    for ts in _recent_trading_days(3):
        db_session.add(_voo(ts))
    await db_session.flush()
    status = await check_vix_staleness(db_session)
    assert status["stale"] is True
    assert "No VIX candles" in status["detail"]


@pytest.mark.asyncio
async def test_fresh_vix_not_stale(db_session):
    days = _recent_trading_days(5)
    for ts in days:
        db_session.add(_voo(ts))
        db_session.add(_vix(ts))
    await db_session.flush()
    status = await check_vix_staleness(db_session)
    assert status["stale"] is False
    assert status["lag_trading_days"] == 0


@pytest.mark.asyncio
async def test_lagging_vix_is_stale(db_session):
    days = _recent_trading_days(settings.VIX_STALENESS_MAX_LAG_DAYS + 5)
    for ts in days:
        db_session.add(_voo(ts))
    # VIX stopped updating: only the oldest candle exists
    db_session.add(_vix(days[0]))
    await db_session.flush()
    status = await check_vix_staleness(db_session)
    assert status["stale"] is True
    assert status["lag_trading_days"] > settings.VIX_STALENESS_MAX_LAG_DAYS
    assert "degraded" in status["detail"]


@pytest.mark.asyncio
async def test_small_lag_within_threshold_not_stale(db_session):
    days = _recent_trading_days(settings.VIX_STALENESS_MAX_LAG_DAYS + 2)
    for ts in days:
        db_session.add(_voo(ts))
    # VIX lags by exactly the allowed number of trading days
    db_session.add(_vix(days[-(settings.VIX_STALENESS_MAX_LAG_DAYS + 1)]))
    await db_session.flush()
    status = await check_vix_staleness(db_session)
    assert status["stale"] is False
    assert status["lag_trading_days"] == settings.VIX_STALENESS_MAX_LAG_DAYS
