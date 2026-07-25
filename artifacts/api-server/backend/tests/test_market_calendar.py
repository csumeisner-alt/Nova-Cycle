"""
Regression tests locking session classification, gap_class thresholds,
and /api/gap_status response shape.

Covers:
  - DST boundaries (March spring-forward, November fall-back)
  - Full NYSE holidays (Good Friday, observed July 4, weekend-observed dates)
  - Half-days (day after Thanksgiving, Christmas Eve, July 3)
  - Fallback classifier
  - gap_class thresholds (none/micro/minor/macro)
  - /api/gap_status keeps existing fields alongside additive ones
"""

from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.models import Base, VooCandle
from database.db import get_db
from ingestion import market_calendar as mc
from ingestion.fetcher import DataFetcher
from routers.data import router as data_router


# ─────────────────────────────────────────────────────────────────────────────
# DST boundaries
# ─────────────────────────────────────────────────────────────────────────────

class TestDST:
    def test_winter_est_regular_open(self):
        # 2026-01-15 14:30 UTC == 09:30 EST → regular
        assert mc.classify_session(datetime(2026, 1, 15, 14, 30)) == (False, "regular", "calendar")

    def test_winter_est_pre_market(self):
        # 14:29 UTC == 09:29 EST → pre_market
        assert mc.classify_session(datetime(2026, 1, 15, 14, 29)) == (True, "pre_market", "calendar")

    def test_spring_forward_monday_edt(self):
        # DST begins 2026-03-08. Monday 2026-03-09: 13:30 UTC == 09:30 EDT → regular
        assert mc.classify_session(datetime(2026, 3, 9, 13, 30)) == (False, "regular", "calendar")

    def test_friday_before_spring_forward_est(self):
        # 2026-03-06 (still EST): 13:30 UTC == 08:30 EST → pre_market
        assert mc.classify_session(datetime(2026, 3, 6, 13, 30)) == (True, "pre_market", "calendar")

    def test_fall_back_monday_est(self):
        # DST ends 2026-11-01. Monday 2026-11-02: 13:30 UTC == 08:30 EST → pre_market
        assert mc.classify_session(datetime(2026, 11, 2, 13, 30)) == (True, "pre_market", "calendar")

    def test_friday_before_fall_back_edt(self):
        # 2026-10-30 (still EDT): 13:30 UTC == 09:30 EDT → regular
        assert mc.classify_session(datetime(2026, 10, 30, 13, 30)) == (False, "regular", "calendar")

    def test_close_boundary_edt(self):
        # 2026-06-15 20:00 UTC == 16:00 EDT → after_hours (close is exclusive)
        assert mc.classify_session(datetime(2026, 6, 15, 20, 0)) == (True, "after_hours", "calendar")
        # 19:55 UTC == 15:55 EDT → regular
        assert mc.classify_session(datetime(2026, 6, 15, 19, 55)) == (False, "regular", "calendar")

    def test_aware_utc_timestamp_supported(self):
        ts = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)  # 10:00 EDT
        assert mc.classify_session(ts) == (False, "regular", "calendar")


# ─────────────────────────────────────────────────────────────────────────────
# Holidays
# ─────────────────────────────────────────────────────────────────────────────

class TestHolidays:
    def test_good_friday(self):
        # Easter 2026 is April 5 → Good Friday April 3
        assert mc.is_market_holiday(date(2026, 4, 3))
        assert not mc.is_trading_day(date(2026, 4, 3))
        # A mid-day candle on Good Friday is extended-hours
        assert mc.classify_session(datetime(2026, 4, 3, 15, 0)) == (True, "after_hours", "calendar")

    def test_july4_observed_friday(self):
        # 2026-07-04 is a Saturday → observed Friday July 3
        assert mc.is_market_holiday(date(2026, 7, 3))
        assert not mc.is_market_holiday(date(2026, 7, 4))  # Saturday itself not in set... actually check membership
        assert mc.classify_session(datetime(2026, 7, 3, 15, 0)) == (True, "after_hours", "calendar")

    def test_july4_observed_monday(self):
        # 2027-07-04 is a Sunday → observed Monday July 5
        assert mc.is_market_holiday(date(2027, 7, 5))

    def test_july4_weekday(self):
        # 2025-07-04 is a Friday → holiday on the day itself
        assert mc.is_market_holiday(date(2025, 7, 4))

    def test_fixed_holidays_2026(self):
        assert mc.is_market_holiday(date(2026, 1, 1))    # New Year's
        assert mc.is_market_holiday(date(2026, 1, 19))   # MLK
        assert mc.is_market_holiday(date(2026, 2, 16))   # Washington's Birthday
        assert mc.is_market_holiday(date(2026, 5, 25))   # Memorial Day
        assert mc.is_market_holiday(date(2026, 6, 19))   # Juneteenth
        assert mc.is_market_holiday(date(2026, 9, 7))    # Labor Day
        assert mc.is_market_holiday(date(2026, 11, 26))  # Thanksgiving
        assert mc.is_market_holiday(date(2026, 12, 25))  # Christmas

    def test_juneteenth_not_before_2022(self):
        assert not mc.is_market_holiday(date(2021, 6, 18))
        assert date(2021, 6, 19) not in mc.market_holidays(2021)

    def test_weekend_not_trading_day(self):
        assert not mc.is_trading_day(date(2026, 7, 25))  # Saturday
        assert mc.classify_session(datetime(2026, 7, 25, 15, 0))[0] is True

    def test_ordinary_weekday_is_trading_day(self):
        assert mc.is_trading_day(date(2026, 7, 22))


# ─────────────────────────────────────────────────────────────────────────────
# Half-days
# ─────────────────────────────────────────────────────────────────────────────

class TestHalfDays:
    def test_day_after_thanksgiving(self):
        # Thanksgiving 2026 = Nov 26 → half-day Nov 27
        assert mc.is_half_day(date(2026, 11, 27))
        # 12:59 EST (17:59 UTC) → regular
        assert mc.classify_session(datetime(2026, 11, 27, 17, 59)) == (False, "regular", "calendar")
        # 13:00 EST (18:00 UTC) → after_hours (early close)
        assert mc.classify_session(datetime(2026, 11, 27, 18, 0)) == (True, "after_hours", "calendar")

    def test_christmas_eve(self):
        # 2026-12-24 is a Thursday → half-day
        assert mc.is_half_day(date(2026, 12, 24))
        # 13:30 EST (18:30 UTC) → after_hours
        assert mc.classify_session(datetime(2026, 12, 24, 18, 30)) == (True, "after_hours", "calendar")

    def test_christmas_eve_weekend_not_half_day(self):
        # 2022-12-24 is a Saturday → not a half-day
        assert not mc.is_half_day(date(2022, 12, 24))

    def test_july3_half_day_when_both_weekdays(self):
        # 2025: Jul 3 Thu, Jul 4 Fri → half-day
        assert mc.is_half_day(date(2025, 7, 3))

    def test_july3_not_half_day_when_observed_holiday(self):
        # 2026: Jul 3 is the observed July-4 holiday, not a half day
        assert not mc.is_half_day(date(2026, 7, 3))

    def test_normal_day_full_close(self):
        # Ordinary day: 15:30 EST in December (20:30 UTC) → regular
        assert mc.classify_session(datetime(2026, 12, 22, 20, 30)) == (False, "regular", "calendar")


# ─────────────────────────────────────────────────────────────────────────────
# Fallback classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestFallback:
    def test_fallback_used_when_calendar_fails(self):
        with patch.object(mc, "is_trading_day", side_effect=RuntimeError("boom")):
            # 14:00 UTC → 10:00 fixed-offset EDT → regular
            is_ext, session, method = mc.classify_session(datetime(2026, 6, 15, 14, 0))
        assert (is_ext, session, method) == (False, "regular", "fallback")

    def test_fallback_boundaries_direct(self):
        assert mc._fallback_classify(datetime(2026, 6, 15, 8, 0)) == (True, "pre_market", "fallback")
        assert mc._fallback_classify(datetime(2026, 6, 15, 13, 30)) == (False, "regular", "fallback")
        assert mc._fallback_classify(datetime(2026, 6, 15, 20, 0)) == (True, "after_hours", "fallback")
        assert mc._fallback_classify(datetime(2026, 6, 15, 2, 0)) == (True, "after_hours", "fallback")

    def test_fetcher_uses_calendar_classifier(self):
        # DataFetcher wrapper drops the method but keeps classification
        is_ext, session = DataFetcher._classify_session(datetime(2026, 6, 15, 14, 0))
        assert (is_ext, session) == (False, "regular")


# ─────────────────────────────────────────────────────────────────────────────
# gap_class thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestGapClass:
    @pytest.mark.parametrize("pct,expected", [
        (0.0, "none"),
        (0.05, "micro"),
        (-0.05, "micro"),
        (0.099, "micro"),
        (0.1, "minor"),     # boundary: not < micro threshold
        (0.5, "minor"),
        (-0.5, "minor"),
        (1.0, "minor"),     # boundary: not > macro threshold
        (1.001, "macro"),
        (-2.5, "macro"),
    ])
    def test_thresholds(self, pct, expected):
        assert DataFetcher.classify_gap_magnitude(pct) == expected

    @pytest.mark.asyncio
    async def test_detect_gap_includes_class_and_momentum(self):
        f = DataFetcher()
        res = await f.detect_gap(prev_close=100.0, premarket_open=101.5)
        assert res["gap_type"] == "gap_up"
        assert res["gap_class"] == "macro"
        assert res["gap_percent"] == 1.5
        assert res["gap_momentum"] is None

    @pytest.mark.asyncio
    async def test_detect_gap_zero_prev_close(self):
        f = DataFetcher()
        res = await f.detect_gap(prev_close=0.0, premarket_open=100.0)
        assert res == {"gap_percent": 0.0, "gap_type": "none",
                       "gap_class": "none", "gap_momentum": None}


# ─────────────────────────────────────────────────────────────────────────────
# /api/gap_status response shape
# ─────────────────────────────────────────────────────────────────────────────

EXISTING_FIELDS = {"ticker", "gap_percent", "gap_type", "timestamp", "session_type"}
ADDITIVE_FIELDS = {"gap_class", "gap_momentum"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    app = FastAPI()
    app.include_router(data_router, prefix="/api")

    async def override_get_db():
        async with session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, session_maker
    await engine.dispose()


@pytest.mark.asyncio
async def test_gap_status_empty_db_shape(client):
    ac, _ = client
    resp = await ac.get("/api/gap_status", params={"ticker": "VOO"})
    assert resp.status_code == 200
    body = resp.json()
    assert EXISTING_FIELDS <= set(body)
    assert ADDITIVE_FIELDS <= set(body)
    assert body["gap_type"] == "none"
    assert body["gap_class"] == "none"
    assert body["gap_momentum"] is None


@pytest.mark.asyncio
async def test_gap_status_with_candle(client):
    ac, session_maker = client
    async with session_maker() as session:
        session.add(VooCandle(
            ticker="VOO", timestamp=datetime(2026, 7, 24, 13, 35),
            open=100.0, high=101.0, low=99.5, close=100.8, volume=1000.0,
            timeframe="5min", is_extended_hours=False, session_type="regular",
            gap_percent=0.5, gap_type="none",
        ))
        await session.commit()

    resp = await ac.get("/api/gap_status", params={"ticker": "VOO"})
    assert resp.status_code == 200
    body = resp.json()
    # Existing fields unchanged
    assert body["ticker"] == "VOO"
    assert body["gap_percent"] == 0.5
    assert body["gap_type"] == "none"
    assert body["session_type"] == "regular"
    assert body["timestamp"] == "2026-07-24T13:35:00"
    # Additive fields present and consistent
    assert body["gap_class"] == "minor"
    assert body["gap_momentum"] is None
    assert body["close"] == 100.8
