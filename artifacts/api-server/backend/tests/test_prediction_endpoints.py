"""End-to-end regression test for the prediction endpoints.

The 0.5-neutral regression was only visible in production logs. This test
spins up the real FastAPI app (no lifespan / scheduler) against an isolated
seeded SQLite database, uses the *committed* model pickles, and asserts that
/api/predict_long, /api/predict_short, and /api/hold_time_estimate return
200s with real (non-fallback) ML confidences — catching any regression that
would silently serve the neutral 0.5 fallback after a deploy.
"""

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.db import get_session
from database.models import Base, VooCandle, VixCandle
from main import app


# ---------------------------------------------------------------------------
# Seed data: enough daily history for sma200 & long features, plus a full
# 5-min session tail for the short model.
# ---------------------------------------------------------------------------
def _daily_candles(n=300):
    idx = pd.bdate_range(end="2026-07-24", periods=n)
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    rows = []
    for i, ts in enumerate(idx):
        c = float(close[i])
        rows.append(VooCandle(
            ticker="VOO", timestamp=ts.to_pydatetime(),
            open=c - 0.2, high=c + 0.5, low=c - 0.5, close=c,
            volume=float(rng.uniform(1e6, 5e6)),
            timeframe="daily", is_extended_hours=False,
            session_type="regular", gap_percent=0.0, gap_type="none",
        ))
    return rows


def _fivemin_candles(n=500):
    idx = pd.date_range(end="2026-07-24 19:55", periods=n, freq="5min")
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.0, 0.15, n))
    rows = []
    for i, ts in enumerate(idx):
        c = float(close[i])
        hour = ts.hour + ts.minute / 60.0
        if hour < 13.5:
            session_type, ext = "pre_market", True
        elif hour >= 20.0:
            session_type, ext = "after_hours", True
        else:
            session_type, ext = "regular", False
        rows.append(VooCandle(
            ticker="VOO", timestamp=ts.to_pydatetime(),
            open=c - 0.05, high=c + 0.1, low=c - 0.1, close=c,
            volume=float(rng.uniform(1e4, 5e4)),
            timeframe="5min", is_extended_hours=ext,
            session_type=session_type, gap_percent=0.0, gap_type="none",
        ))
    return rows


def _vix_candles(n=300):
    idx = pd.bdate_range(end="2026-07-24", periods=n)
    rng = np.random.default_rng(3)
    close = np.clip(16 + np.cumsum(rng.normal(0, 0.3, n)), 10, 30)
    return [
        VixCandle(
            ticker="^VIX", timestamp=ts.to_pydatetime(),
            open=float(c), high=float(c) + 0.5, low=float(c) - 0.5,
            close=float(c), volume=0.0, timeframe="daily",
        )
        for ts, c in zip(idx, close)
    ]


# ---------------------------------------------------------------------------
# App fixture: isolated tmp SQLite DB + dependency override (no lifespan,
# so the scheduler / ingestion pipeline never runs).
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def client(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        session.add_all(_daily_candles() + _fivemin_candles() + _vix_candles())
        await session.commit()

    async def _override():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = _override
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_session, None)
        await engine.dispose()


class TestPredictionEndpointsE2E:
    """Committed models + seeded candles must yield real, non-fallback scores."""

    async def test_predict_long_returns_real_confidence(self, client):
        resp = await client.post("/api/predict_long", params={"ticker": "VOO"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("note") is None, f"neutral no-data response: {body}"
        assert body["ml_fallback"] is False, f"served neutral fallback: {body}"
        assert body["ml_confidence"] != 0.5
        assert 0.0 <= body["ml_confidence"] <= 1.0
        assert body["signal"] in ("buy", "sell", "neutral")

    async def test_predict_short_returns_real_confidence(self, client):
        resp = await client.post("/api/predict_short", params={"ticker": "VOO"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("note") is None, f"neutral no-data response: {body}"
        assert body["ml_fallback"] is False, f"served neutral fallback: {body}"
        assert body["ml_confidence"] != 0.5
        assert 0.0 <= body["ml_confidence"] <= 1.0
        assert body["signal"] in ("buy", "sell", "neutral")

    async def test_hold_time_estimate_returns_200(self, client):
        # Prime the in-process caches the same way production traffic does.
        assert (await client.post("/api/predict_long")).status_code == 200
        assert (await client.post("/api/predict_short")).status_code == 200

        resp = await client.post("/api/hold_time_estimate", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ticker"] == "VOO"

    async def test_full_sequence_no_fallbacks_recorded(self, client):
        """The healthz fallback counters must not grow during a healthy run."""
        from routers.predictions import _ml_fallback_stats

        before = {k: v["count"] for k, v in _ml_fallback_stats.items()}
        await client.post("/api/predict_long")
        await client.post("/api/predict_short")
        after = {k: v["count"] for k, v in _ml_fallback_stats.items()}
        assert after == before, f"fallbacks recorded during healthy run: {after}"
