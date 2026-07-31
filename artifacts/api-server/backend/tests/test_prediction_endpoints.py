"""End-to-end regression test for the prediction endpoints.

The 0.5-neutral regression was only visible in production logs. This test
spins up the real FastAPI app (no lifespan / scheduler) against an isolated
seeded SQLite database, uses the *committed* model pickles, and asserts that
/api/predict_long, /api/predict_short, and /api/hold_time_estimate return
200s with real (non-fallback) ML confidences — catching any regression that
would silently serve the neutral 0.5 fallback after a deploy.
"""

import datetime
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


# ---------------------------------------------------------------------------
# Spike-quarantine propagation: a spiked daily candle must surface as
# data_quality_degraded=True in the /api/predict_long JSON response.
# ---------------------------------------------------------------------------

def _daily_candles_with_spike(n=300):
    """300 valid daily candles followed by one with high < open (July 30 shape).

    The spiked candle is stored directly in the DB via the ORM (bypassing ingest
    validation), simulating a stored glitch that _drop_invalid_ohlc must catch at
    prediction time.
    """
    valid = _daily_candles(n)
    # Add a spiked candle one business day after the last valid candle.
    last_ts = valid[-1].timestamp
    spike_ts = last_ts + datetime.timedelta(days=1)
    spiked = VooCandle(
        ticker="VOO",
        timestamp=spike_ts,
        open=680.12,
        high=676.71,   # high < open — cross-bar spike shape
        low=675.58,
        close=681.55,
        volume=1_000_000.0,
        timeframe="daily",
        is_extended_hours=False,
        session_type="regular",
        gap_percent=0.0,
        gap_type="none",
    )
    return valid + [spiked]


@pytest_asyncio.fixture
async def client_with_spike(tmp_path):
    """FastAPI test client seeded with a spiked daily candle at the end."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'spike_test.db'}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        session.add_all(
            _daily_candles_with_spike() + _fivemin_candles() + _vix_candles()
        )
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


class TestPredictLongSpikeQuarantinePropagation:
    """End-to-end: a quarantined daily spike must appear in the API response."""

    async def test_data_quality_degraded_is_true(self, client_with_spike):
        """predict_long returns data_quality_degraded=True when a spiked candle
        is quarantined by _drop_invalid_ohlc at prediction time."""
        resp = await client_with_spike.post("/api/predict_long", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("data_quality_degraded") is True, (
            f"expected data_quality_degraded=True but got: {body}"
        )

    async def test_data_quality_reason_is_non_empty(self, client_with_spike):
        """The reason string must describe which candle was quarantined."""
        resp = await client_with_spike.post("/api/predict_long", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        reason = body.get("data_quality_reason", "")
        assert reason, (
            f"expected non-empty data_quality_reason but got empty string; full body: {body}"
        )
        # The reason should mention the quarantine so callers know what happened.
        assert "quarantine" in reason.lower() or "high_below" in reason.lower(), (
            f"reason does not describe the OHLC violation: {reason!r}"
        )
