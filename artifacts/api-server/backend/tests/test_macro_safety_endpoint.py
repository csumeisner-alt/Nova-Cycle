"""
Integration test: /api/macro_safety thresholds reflect the ±65 config values
==============================================================================
Regression guard: previously LONG_STRONG_BULL/BEAR were hardcoded to ±70 in
macro_override.py while the long-gauge BUY/SELL signals fired at ±65.  Now they
are driven by ``settings.LONG_BUY_THRESHOLD`` / ``settings.LONG_SELL_THRESHOLD``.

This test hits the live endpoint that operators use and asserts the ``thresholds``
block in the response carries the current config values — not a stale hardcode —
so operators are never shown misleading suppression boundaries.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.db import get_db, get_session
from database.models import Base, VixCandle, VooCandle
from main import app


# ---------------------------------------------------------------------------
# Minimal seed data: the macro_safety endpoint needs at least one VIX candle
# to compute vix_regime without errors; VOO candles keep DB consistent.
# ---------------------------------------------------------------------------

def _seed_vix_candles(n: int = 5) -> list[VixCandle]:
    idx = pd.bdate_range(end="2026-07-25", periods=n)
    return [
        VixCandle(
            ticker="^VIX",
            timestamp=ts.to_pydatetime(),
            open=17.0, high=18.0, low=16.0, close=17.5,
            volume=0.0, timeframe="daily",
        )
        for ts in idx
    ]


def _seed_voo_candles(n: int = 5) -> list[VooCandle]:
    idx = pd.bdate_range(end="2026-07-25", periods=n)
    rng = np.random.default_rng(99)
    close = 100 + np.cumsum(rng.normal(0.05, 0.5, n))
    rows = []
    for i, ts in enumerate(idx):
        c = float(close[i])
        rows.append(VooCandle(
            ticker="VOO",
            timestamp=ts.to_pydatetime(),
            open=c - 0.1, high=c + 0.3, low=c - 0.3, close=c,
            volume=1_000_000.0,
            timeframe="daily",
            is_extended_hours=False,
            session_type="regular",
            gap_percent=0.0,
            gap_type="none",
        ))
    return rows


# ---------------------------------------------------------------------------
# App fixture: isolated tmp SQLite DB, no lifespan / scheduler.
# Both get_db and get_session are overridden so every router finds the same DB.
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def macro_client(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'test_macro_safety.db'}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        session.add_all(_seed_vix_candles() + _seed_voo_candles())
        await session.commit()

    async def _override():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = _override
    app.dependency_overrides[get_session] = _override
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_session, None)
        await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMacroSafetyEndpointThresholds:
    """
    The operator health endpoint must surface the config-driven ±65 thresholds,
    not an older hardcoded value.
    """

    async def test_endpoint_returns_200(self, macro_client):
        resp = await macro_client.get("/api/macro_safety", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text

    async def test_long_strong_bull_equals_config_long_buy_threshold(self, macro_client):
        resp = await macro_client.get("/api/macro_safety", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        returned_bull = body["thresholds"]["long_strong_bull"]
        assert returned_bull == settings.LONG_BUY_THRESHOLD, (
            f"Endpoint returned long_strong_bull={returned_bull!r} but "
            f"settings.LONG_BUY_THRESHOLD={settings.LONG_BUY_THRESHOLD!r}. "
            "macro_override.py may have a stale hardcode."
        )

    async def test_long_strong_bear_equals_config_long_sell_threshold(self, macro_client):
        resp = await macro_client.get("/api/macro_safety", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        returned_bear = body["thresholds"]["long_strong_bear"]
        assert returned_bear == settings.LONG_SELL_THRESHOLD, (
            f"Endpoint returned long_strong_bear={returned_bear!r} but "
            f"settings.LONG_SELL_THRESHOLD={settings.LONG_SELL_THRESHOLD!r}. "
            "macro_override.py may have a stale hardcode."
        )

    async def test_thresholds_are_symmetric(self, macro_client):
        resp = await macro_client.get("/api/macro_safety", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        bull = body["thresholds"]["long_strong_bull"]
        bear = body["thresholds"]["long_strong_bear"]
        assert bull == -bear, (
            f"Thresholds should be symmetric; got bull={bull}, bear={bear}."
        )

    async def test_thresholds_are_65_not_70(self, macro_client):
        """Explicit regression: the old hardcoded value was ±70; must now be ±65."""
        resp = await macro_client.get("/api/macro_safety", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        bull = body["thresholds"]["long_strong_bull"]
        bear = body["thresholds"]["long_strong_bear"]
        assert bull != 70.0, (
            "long_strong_bull is still 70.0 — the stale hardcode was not removed."
        )
        assert bear != -70.0, (
            "long_strong_bear is still -70.0 — the stale hardcode was not removed."
        )
        assert bull == 65.0, f"Expected long_strong_bull=65.0, got {bull}"
        assert bear == -65.0, f"Expected long_strong_bear=-65.0, got {bear}"

    async def test_response_schema_has_required_operator_fields(self, macro_client):
        """The operator dashboard response must include the full threshold block."""
        resp = await macro_client.get("/api/macro_safety", params={"ticker": "VOO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        thresholds = body.get("thresholds", {})
        assert "long_strong_bull" in thresholds, "Missing long_strong_bull in thresholds"
        assert "long_strong_bear" in thresholds, "Missing long_strong_bear in thresholds"
        assert "ml_override_threshold" in thresholds, "Missing ml_override_threshold in thresholds"
        # Top-level fields operators rely on
        for field in ("long_score", "suppresses_short_buy", "suppresses_short_sell", "reason"):
            assert field in body, f"Missing operator field: {field}"
