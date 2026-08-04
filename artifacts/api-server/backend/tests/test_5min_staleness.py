"""Tests for the VOO 5-min feed staleness check (quiet intraday stall detection)."""

from datetime import date, datetime, timedelta

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


# ── Off-hours: weekend and holiday ────────────────────────────────────────────

def _et_to_utc_naive(dt_et: datetime) -> datetime:
    """Convert an ET-aware datetime to a UTC-naive datetime (matching DB convention)."""
    return dt_et.astimezone(market_calendar.timezone.utc).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_saturday_morning_not_stale(db_session):
    """Last bar from Friday close; now = Saturday 10:00 ET → no alarm expected."""
    # Use a known recent Friday: 2025-01-03 is a Friday (trading day).
    friday = datetime(2025, 1, 3, 16, 0, tzinfo=market_calendar.EASTERN)  # 4 PM ET close
    saturday = datetime(2025, 1, 4, 10, 0, tzinfo=market_calendar.EASTERN)  # Sat morning

    db_session.add(_bar(_et_to_utc_naive(friday)))
    await db_session.flush()

    now_utc = _et_to_utc_naive(saturday)
    status = await check_5min_staleness(db_session, now=now_utc)

    assert status["market_open"] is False, "Saturday should not be market_open"
    assert status["stale"] is False, "Should not alarm on a weekend morning"


@pytest.mark.asyncio
async def test_market_holiday_morning_not_stale(db_session):
    """Last bar from the day before a holiday; now = holiday morning → no alarm expected.

    2025-01-01 is New Year's Day (NYSE holiday). The prior trading day is
    2024-12-31 (Tuesday). A bar stored at the 31st's close should not trigger
    staleness when the check is run on holiday morning (Jan 1 at 10:00 ET).
    """
    prior_close = datetime(2024, 12, 31, 16, 0, tzinfo=market_calendar.EASTERN)
    holiday_morning = datetime(2025, 1, 1, 10, 0, tzinfo=market_calendar.EASTERN)

    assert market_calendar.is_market_holiday(holiday_morning.date()), (
        "2025-01-01 must be a recognised NYSE holiday for this test to be valid"
    )

    db_session.add(_bar(_et_to_utc_naive(prior_close)))
    await db_session.flush()

    now_utc = _et_to_utc_naive(holiday_morning)
    status = await check_5min_staleness(db_session, now=now_utc)

    assert status["market_open"] is False, "Holiday should not be market_open"
    assert status["stale"] is False, "Should not alarm on a market-holiday morning"


# ── Half-day (early close at 13:00 ET) ───────────────────────────────────────

@pytest.mark.asyncio
async def test_half_day_after_early_close_not_stale(db_session):
    """Last bar at 12:55 ET on a half-day; check at 14:00 ET → no alarm.

    The day after Thanksgiving (2024-11-29) is a well-known NYSE half-day
    where the regular session ends at 13:00 ET instead of 16:00 ET.
    After 13:00 ET the market is closed so the staleness check should
    never alarm regardless of how old the last bar is.
    """
    # 2024-11-29 = day after Thanksgiving → confirmed half-day
    half_day = date(2024, 11, 29)
    assert market_calendar.is_half_day(half_day), (
        "2024-11-29 must be recognised as a half-day for this test to be valid"
    )
    assert market_calendar.is_trading_day(half_day), (
        "2024-11-29 must be a trading day for this test to be valid"
    )

    # Last bar stored at 12:55 ET (5 min before the early 13:00 close)
    last_bar_et = datetime(2024, 11, 29, 12, 55, tzinfo=market_calendar.EASTERN)
    last_bar_utc = _et_to_utc_naive(last_bar_et)
    db_session.add(_bar(last_bar_utc))
    await db_session.flush()

    # Staleness check run at 14:00 ET — one full hour after the early close
    check_et = datetime(2024, 11, 29, 14, 0, tzinfo=market_calendar.EASTERN)
    now_utc = _et_to_utc_naive(check_et)

    status = await check_5min_staleness(db_session, now=now_utc)

    assert status["market_open"] is False, (
        "Market should be closed at 14:00 ET on a half-day (closes 13:00 ET)"
    )
    assert status["stale"] is False, (
        "Should not alarm after the early close on a half-day"
    )
