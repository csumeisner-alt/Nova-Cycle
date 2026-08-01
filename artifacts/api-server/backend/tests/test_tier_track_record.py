"""
Tier track record tests
=======================
Covers the pure aggregation (compute_tier_track_record), sparse-sample
handling, price-data-absent exclusion, and the /api/tier_track_record
endpoint (window validation + end-to-end aggregation from SignalHistory).
"""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)

from performance_engine import (
    MIN_TIER_SAMPLE,
    compute_tier_track_record,
    get_tier_track_record,
)
from database.models import Base, SignalHistory, VooCandle
from main import app
from database.db import get_session


def _cycle(tier, ret, absent=False):
    return {
        "conviction_tier_at_buy": tier,
        "return_percent": ret,
        "price_data_absent": absent,
    }


class TestComputeTierTrackRecord:
    def test_empty_cycles_safe_shape(self):
        out = compute_tier_track_record([])
        assert out["overall"]["trade_count"] == 0
        assert out["overall"]["win_rate"] is None
        assert out["overall"]["sufficient_sample"] is False
        for key in ("high_conviction", "opportunity", "untiered"):
            assert out["tiers"][key]["trade_count"] == 0
            assert out["tiers"][key]["win_rate"] is None
        assert out["excluded_price_data_absent"] == 0
        assert out["min_sample_size"] == MIN_TIER_SAMPLE

    def test_per_tier_win_rate_and_avg_return(self):
        cycles = (
            [_cycle("high_conviction", 1.0)] * 4
            + [_cycle("high_conviction", -0.5)]
            + [_cycle("opportunity", -1.0)] * 3
            + [_cycle("opportunity", 2.0)] * 2
        )
        out = compute_tier_track_record(cycles)
        hc = out["tiers"]["high_conviction"]
        assert hc["trade_count"] == 5
        assert hc["win_rate"] == pytest.approx(4 / 5)
        assert hc["avg_return_percent"] == pytest.approx((4 * 1.0 - 0.5) / 5)
        opp = out["tiers"]["opportunity"]
        assert opp["win_rate"] == pytest.approx(2 / 5)
        assert out["overall"]["trade_count"] == 10
        assert out["overall"]["sufficient_sample"] is True

    def test_small_sample_reports_null_not_percentage(self):
        cycles = [_cycle("high_conviction", 5.0)] * (MIN_TIER_SAMPLE - 1)
        out = compute_tier_track_record(cycles)
        hc = out["tiers"]["high_conviction"]
        assert hc["trade_count"] == MIN_TIER_SAMPLE - 1
        assert hc["win_rate"] is None
        assert hc["avg_return_percent"] is None
        assert hc["sufficient_sample"] is False

    def test_null_tier_grouped_as_untiered(self):
        cycles = [_cycle(None, 1.0)] * MIN_TIER_SAMPLE
        out = compute_tier_track_record(cycles)
        assert out["tiers"]["untiered"]["trade_count"] == MIN_TIER_SAMPLE
        assert out["tiers"]["untiered"]["win_rate"] == 1.0

    def test_unknown_tier_value_grouped_as_untiered(self):
        out = compute_tier_track_record([_cycle("weird_tier", 1.0)])
        assert out["tiers"]["untiered"]["trade_count"] == 1

    def test_price_data_absent_cycles_excluded(self):
        cycles = (
            [_cycle("high_conviction", 1.0)] * MIN_TIER_SAMPLE
            + [_cycle("high_conviction", 0.0, absent=True)] * 3
        )
        out = compute_tier_track_record(cycles)
        assert out["excluded_price_data_absent"] == 3
        hc = out["tiers"]["high_conviction"]
        assert hc["trade_count"] == MIN_TIER_SAMPLE
        assert hc["win_rate"] == 1.0
        assert out["overall"]["trade_count"] == MIN_TIER_SAMPLE


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    async def _override():
        yield db_session

    app.dependency_overrides[get_session] = _override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.pop(get_session, None)


def _signal(ts, sig_type, tier, conf=0.8):
    return SignalHistory(
        timestamp=ts,
        ticker="VOO",
        signal_type=sig_type,
        gauge_type="long",
        confidence=conf,
        session_type="regular",
        conviction_tier=tier,
    )


def _candle(ts, close):
    return VooCandle(
        ticker="VOO", timestamp=ts, timeframe="daily",
        open=close, high=close + 1, low=close - 1, close=close,
        volume=1000.0, is_extended_hours=False, session_type="regular",
    )


@pytest.mark.asyncio
async def test_endpoint_rejects_bad_window(client):
    res = await client.get("/api/tier_track_record?ticker=VOO&window=7d")
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_rejects_bad_ticker(client):
    res = await client.get("/api/tier_track_record?ticker=SPY")
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_empty_db_safe_shape(client):
    res = await client.get("/api/tier_track_record?ticker=VOO&window=30d")
    assert res.status_code == 200
    body = res.json()
    assert body["ticker"] == "VOO"
    assert body["window"] == "30d"
    assert body["available_windows"] == ["30d", "90d", "all"]
    assert body["overall"]["trade_count"] == 0
    assert body["tiers"]["high_conviction"]["win_rate"] is None


@pytest.mark.asyncio
async def test_endpoint_aggregates_completed_cycles_per_tier(
    client, db_session: AsyncSession
):
    """Winning high-conviction cycles + a losing opportunity cycle end up
    in the right tier buckets, computed from stored signals and candles."""
    base = datetime.utcnow() - timedelta(days=5)
    price = 100.0
    # Enough alternating buy/sell pairs for a sufficient high-conviction sample
    for i in range(MIN_TIER_SAMPLE):
        buy_ts = base + timedelta(hours=4 * i)
        sell_ts = buy_ts + timedelta(hours=2)
        db_session.add(_candle(buy_ts - timedelta(minutes=5), price))
        db_session.add(_candle(sell_ts - timedelta(minutes=5), price + 1.0))
        db_session.add(_signal(buy_ts, "buy", "high_conviction"))
        db_session.add(_signal(sell_ts, "sell", "high_conviction"))
        price += 2.0
    await db_session.commit()

    res = await client.get("/api/tier_track_record?ticker=VOO&window=30d")
    assert res.status_code == 200
    body = res.json()
    hc = body["tiers"]["high_conviction"]
    assert hc["trade_count"] == MIN_TIER_SAMPLE
    assert hc["sufficient_sample"] is True
    assert hc["win_rate"] == 1.0
    assert hc["avg_return_percent"] > 0.0


@pytest.mark.asyncio
async def test_get_tier_track_record_all_window(db_session: AsyncSession):
    out = await get_tier_track_record(db_session, ticker="VOO", window="all")
    assert out["window"] == "all"
    assert out["overall"]["trade_count"] == 0
