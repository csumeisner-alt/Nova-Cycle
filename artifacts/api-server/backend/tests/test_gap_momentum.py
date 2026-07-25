"""
Tests: gap follow-through (gap_momentum) correctness.

Locks in:
  - Sign convention: positive = follow-through (price continued in the gap's
    direction), negative = fade (price moved against the gap).
  - Null cases: zero gap, missing/short post-open candle window, zero or NaN
    open in the first candle.
  - Read-time DB path: /api/gap_status returns null vs numeric gap_momentum
    depending on what's in the database.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.db import get_db
from database.models import Base, VooCandle
from ingestion.fetcher import DataFetcher
from routers.data import router as data_router

N = DataFetcher.GAP_MOMENTUM_CANDLES


def make_candles(opens: list[float], closes: list[float]) -> pd.DataFrame:
    """Build a post-open 5-min candle frame (oldest first)."""
    idx = pd.date_range("2026-07-24 13:30", periods=len(opens), freq="5min")
    return pd.DataFrame({"open": opens, "close": closes}, index=idx)


def flat_candles(n: int = N, open_1: float = 100.0, close_n: float = 100.0):
    opens = [open_1] + [100.0] * (n - 1)
    closes = [100.0] * (n - 1) + [close_n]
    return make_candles(opens, closes)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: compute_gap_momentum
# ─────────────────────────────────────────────────────────────────────────────

class TestSignConvention:
    def test_gap_up_price_rises_is_positive_follow_through(self):
        # gap up, price rises 100 → 101 over first 30 min → +1.0
        m = DataFetcher.compute_gap_momentum(1.5, flat_candles(close_n=101.0))
        assert m == pytest.approx(1.0)

    def test_gap_up_price_falls_is_negative_fade(self):
        m = DataFetcher.compute_gap_momentum(1.5, flat_candles(close_n=99.0))
        assert m == pytest.approx(-1.0)

    def test_gap_down_price_falls_is_positive_follow_through(self):
        # gap down and price keeps falling → follow-through → positive
        m = DataFetcher.compute_gap_momentum(-1.5, flat_candles(close_n=99.0))
        assert m == pytest.approx(1.0)

    def test_gap_down_price_rises_is_negative_fade(self):
        m = DataFetcher.compute_gap_momentum(-1.5, flat_candles(close_n=101.0))
        assert m == pytest.approx(-1.0)

    def test_uses_first_open_and_nth_close_only(self):
        # Extra candles beyond N and noisy middle values must not matter.
        opens = [100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 999.0]
        closes = [7.0, 8.0, 9.0, 10.0, 11.0, 102.0, 999.0]
        m = DataFetcher.compute_gap_momentum(2.0, make_candles(opens, closes))
        assert m == pytest.approx(2.0)

    def test_sorts_candles_by_index(self):
        df = flat_candles(close_n=101.0).sort_index(ascending=False)
        m = DataFetcher.compute_gap_momentum(1.0, df)
        assert m == pytest.approx(1.0)

    def test_rounded_to_4_decimals(self):
        m = DataFetcher.compute_gap_momentum(1.0, flat_candles(open_1=3.0, close_n=100.0))
        assert m == round(m, 4)


class TestNullCases:
    def test_zero_gap_returns_none(self):
        assert DataFetcher.compute_gap_momentum(0.0, flat_candles()) is None

    def test_none_candles_returns_none(self):
        assert DataFetcher.compute_gap_momentum(1.0, None) is None

    def test_empty_candles_returns_none(self):
        assert DataFetcher.compute_gap_momentum(1.0, pd.DataFrame()) is None

    def test_fewer_than_required_candles_returns_none(self):
        df = flat_candles()[: N - 1]
        assert DataFetcher.compute_gap_momentum(1.0, df) is None

    def test_exactly_required_candles_is_numeric(self):
        assert DataFetcher.compute_gap_momentum(1.0, flat_candles()) is not None

    def test_zero_open_returns_none(self):
        assert DataFetcher.compute_gap_momentum(1.0, flat_candles(open_1=0.0)) is None

    def test_nan_open_returns_none(self):
        assert DataFetcher.compute_gap_momentum(1.0, flat_candles(open_1=np.nan)) is None

    def test_nan_close_returns_none(self):
        assert DataFetcher.compute_gap_momentum(1.0, flat_candles(close_n=np.nan)) is None

    def test_malformed_columns_returns_none_never_raises(self):
        df = pd.DataFrame({"foo": range(N)})
        assert DataFetcher.compute_gap_momentum(1.0, df) is None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint tests: /api/gap_status read-time DB path
# ─────────────────────────────────────────────────────────────────────────────

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


def five_min_candle(ts, session_type, open_=100.0, close=100.0, gap_percent=0.0):
    return VooCandle(
        ticker="VOO", timestamp=ts, open=open_, high=open_ + 1, low=open_ - 1,
        close=close, volume=1000.0, timeframe="5min",
        is_extended_hours=session_type != "regular", session_type=session_type,
        gap_percent=gap_percent, gap_type="none",
    )


async def seed(session_maker, candles):
    async with session_maker() as session:
        session.add_all(candles)
        await session.commit()


@pytest.mark.asyncio
async def test_gap_status_no_gap_candle_returns_null_momentum(client):
    ac, session_maker = client
    ts = datetime(2026, 7, 24, 13, 35)
    await seed(session_maker, [five_min_candle(ts, "regular")])
    body = (await ac.get("/api/gap_status", params={"ticker": "VOO"})).json()
    assert body["gap_momentum"] is None


@pytest.mark.asyncio
async def test_gap_status_gap_but_too_few_post_open_candles_is_null(client):
    ac, session_maker = client
    day = datetime(2026, 7, 24)
    candles = [five_min_candle(day.replace(hour=9), "pre_market", gap_percent=1.5)]
    for i in range(N - 1):  # one short of required
        candles.append(
            five_min_candle(day.replace(hour=13, minute=30) + timedelta(minutes=5 * i), "regular")
        )
    await seed(session_maker, candles)
    body = (await ac.get("/api/gap_status", params={"ticker": "VOO"})).json()
    assert body["gap_momentum"] is None


@pytest.mark.asyncio
async def test_gap_status_returns_numeric_momentum(client):
    ac, session_maker = client
    day = datetime(2026, 7, 24)
    candles = [five_min_candle(day.replace(hour=9), "pre_market", gap_percent=1.5)]
    # 6 regular candles: first opens at 100, sixth closes at 101 → +1.0
    for i in range(N):
        close = 101.0 if i == N - 1 else 100.0
        candles.append(
            five_min_candle(
                day.replace(hour=13, minute=30) + timedelta(minutes=5 * i),
                "regular", open_=100.0, close=close,
            )
        )
    await seed(session_maker, candles)
    body = (await ac.get("/api/gap_status", params={"ticker": "VOO"})).json()
    assert body["gap_momentum"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_gap_status_stale_gap_from_earlier_day_is_null(client):
    """A gap from a previous day must not produce momentum for today's status."""
    ac, session_maker = client
    old_day = datetime(2026, 7, 17)  # a week earlier: gap + full post-open window
    candles = [five_min_candle(old_day.replace(hour=9), "pre_market", gap_percent=1.5)]
    for i in range(N):
        close = 101.0 if i == N - 1 else 100.0
        candles.append(
            five_min_candle(
                old_day.replace(hour=13, minute=30) + timedelta(minutes=5 * i),
                "regular", open_=100.0, close=close,
            )
        )
    # Today: latest candle has no gap
    today = datetime(2026, 7, 24)
    candles.append(five_min_candle(today.replace(hour=13, minute=35), "regular"))
    await seed(session_maker, candles)
    body = (await ac.get("/api/gap_status", params={"ticker": "VOO"})).json()
    assert body["gap_percent"] == 0.0
    assert body["gap_momentum"] is None


@pytest.mark.asyncio
async def test_gap_status_gap_down_fade_is_negative(client):
    ac, session_maker = client
    day = datetime(2026, 7, 24)
    candles = [five_min_candle(day.replace(hour=9), "pre_market", gap_percent=-1.5)]
    for i in range(N):
        close = 101.0 if i == N - 1 else 100.0  # price rises against gap-down
        candles.append(
            five_min_candle(
                day.replace(hour=13, minute=30) + timedelta(minutes=5 * i),
                "regular", open_=100.0, close=close,
            )
        )
    await seed(session_maker, candles)
    body = (await ac.get("/api/gap_status", params={"ticker": "VOO"})).json()
    assert body["gap_momentum"] == pytest.approx(-1.0)
