"""Tests for the spec-gap endpoints (POST /api/ingest, GET /api/macro_safety,
GET /api/extended_hours) and the notification reliability gate."""

import pytest
from httpx import ASGITransport, AsyncClient
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from main import app
from database.db import get_db
from database.models import Base, VooCandle


def _client():
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


@pytest.mark.asyncio
async def test_price_snapshot_separates_current_and_model_input_prices(tmp_path):
    """The chart contract exposes the freshest price and both model inputs."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'price_snapshot.db'}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    base = datetime(2026, 7, 31, 17, 0)
    async with factory() as session:
        session.add_all([
            VooCandle(
                ticker="VOO", timeframe="daily", timestamp=base,
                open=100, high=103, low=99, close=102, volume=1000,
                is_extended_hours=False, session_type="regular",
            ),
            VooCandle(
                ticker="VOO", timeframe="5min", timestamp=base + timedelta(minutes=5),
                open=104, high=105, low=103, close=104.5, volume=100,
                is_extended_hours=False, session_type="regular",
            ),
        # The current price ignores a zero-volume glitch, but the short model
        # input must match predict_short, which keeps that bar for features.
            VooCandle(
                ticker="VOO", timeframe="5min", timestamp=base + timedelta(minutes=10),
                open=106, high=106, low=106, close=106, volume=0,
                is_extended_hours=False, session_type="regular",
            ),
            VooCandle(
                ticker="VOO", timeframe="5min", timestamp=base - timedelta(days=1),
                open=101, high=102, low=100, close=101, volume=100,
                is_extended_hours=False, session_type="regular",
            ),
        ])
        await session.commit()

    async def override_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        async with _client() as client:
            response = await client.get("/api/price_snapshot", params={"ticker": "VOO"})
        assert response.status_code == 200
        body = response.json()
        assert body["current_price"] == 104.5
        assert body["short_model_price"] == 106.0
        assert body["long_model_price"] == 102.0
        assert body["short_model_timestamp"] == "2026-07-31T17:10:00"
        assert body["reference_price"] == 101.0
        assert body["day_change_percent"] == pytest.approx(3.4653)
        assert body["day_direction"] == "up"
        assert body["current_session"] == "regular"
        assert body["is_extended_hours"] is False
    finally:
        app.dependency_overrides.pop(get_db, None)
        await engine.dispose()


@pytest.mark.asyncio
async def test_price_snapshot_uses_extended_hours_quote_and_prior_close(tmp_path):
    """Premarket/after-hours rows remain eligible for the visible quote."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'extended_price_snapshot.db'}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    base = datetime(2026, 7, 31, 12, 0)  # 08:00 ET premarket
    async with factory() as session:
        session.add_all([
            VooCandle(
                ticker="VOO", timeframe="5min", timestamp=base,
                open=103, high=104, low=103, close=103.5, volume=50,
                is_extended_hours=True, session_type="pre_market",
            ),
            VooCandle(
                ticker="VOO", timeframe="5min",
                timestamp=datetime(2026, 7, 30, 19, 55),
                open=102, high=103, low=101, close=102, volume=100,
                is_extended_hours=False, session_type="regular",
            ),
        ])
        await session.commit()

    async def override_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        async with _client() as client:
            response = await client.get("/api/price_snapshot", params={"ticker": "VOO"})
        assert response.status_code == 200
        body = response.json()
        assert body["current_price"] == 103.5
        assert body["current_session"] == "pre_market"
        assert body["is_extended_hours"] is True
        assert body["reference_price"] == 102.0
        assert body["day_direction"] == "up"
    finally:
        app.dependency_overrides.pop(get_db, None)
        await engine.dispose()


# ── /api/macro_safety ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_macro_safety_shape():
    async with _client() as client:
        resp = await client.get("/api/macro_safety")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "vix_close", "vix_regime", "long_score", "override_active",
        "suppresses_short_buy", "suppresses_short_sell", "reason", "thresholds",
    ):
        assert key in body, f"missing key: {key}"
    assert isinstance(body["override_active"], bool)
    t = body["thresholds"]
    assert t["long_strong_bear"] == -70.0
    assert t["long_strong_bull"] == 70.0
    assert t["ml_override_threshold"] == 0.80


@pytest.mark.asyncio
async def test_macro_safety_override_active_when_bearish(monkeypatch):
    from routers import predictions as pred

    monkeypatch.setattr(pred, "_last_long_score", -85.0)
    async with _client() as client:
        resp = await client.get("/api/macro_safety")
    body = resp.json()
    assert body["override_active"] is True
    assert body["suppresses_short_buy"] is True
    assert body["suppresses_short_sell"] is False
    assert "bear" in body["reason"].lower()


@pytest.mark.asyncio
async def test_macro_safety_inactive_in_neutral_zone(monkeypatch):
    from routers import predictions as pred

    monkeypatch.setattr(pred, "_last_long_score", 10.0)
    async with _client() as client:
        resp = await client.get("/api/macro_safety")
    body = resp.json()
    assert body["override_active"] is False


@pytest.mark.asyncio
async def test_macro_safety_rejects_bad_ticker():
    async with _client() as client:
        resp = await client.get("/api/macro_safety", params={"ticker": "AAPL"})
    assert resp.status_code in (400, 404, 422)


# ── /api/extended_hours ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extended_hours_shape():
    async with _client() as client:
        resp = await client.get("/api/extended_hours", params={"window": "7d"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["window"] == "7d"
    assert isinstance(body["candles"], list)
    assert isinstance(body["session_markers"], list)
    assert body["count"] == len(body["candles"])
    for c in body["candles"]:
        assert c["is_extended_hours"] is True
    for m in body["session_markers"]:
        assert m["from_session"] != m["to_session"]
        assert m["timestamp"] is not None


@pytest.mark.asyncio
async def test_extended_hours_bad_window_falls_back_to_default():
    # _parse_window is intentionally lenient across all data endpoints:
    # unparseable windows fall back to the 30-day default instead of erroring.
    async with _client() as client:
        resp = await client.get("/api/extended_hours", params={"window": "banana"})
    assert resp.status_code == 200
    assert resp.json()["window"] == "banana"


# ── POST /api/ingest ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_manual_ingest_returns_summary(monkeypatch):
    import main

    calls = []

    async def fake_update(db_session):
        calls.append(1)

    monkeypatch.setattr(main.pipeline, "run_incremental_update", fake_update)
    async with _client() as client:
        resp = await client.post("/api/ingest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert calls == [1]
    assert "new_candles" in body and "total_candles" in body
    assert set(body["new_candles"]) == {"daily", "5min"}


@pytest.mark.asyncio
async def test_manual_ingest_failure_returns_500(monkeypatch):
    import main

    async def boom(db_session):
        raise RuntimeError("provider down")

    monkeypatch.setattr(main.pipeline, "run_incremental_update", boom)
    async with _client() as client:
        resp = await client.post("/api/ingest")
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_manual_ingest_concurrent_requests_one_wins(monkeypatch):
    """Two concurrent triggers from an unlocked state: exactly one runs (200)
    and the other is rejected fail-fast (409) — never queued back-to-back."""
    import asyncio

    import main

    runs = []

    async def slow_update(db_session):
        runs.append(1)
        await asyncio.sleep(0.2)

    monkeypatch.setattr(main.pipeline, "run_incremental_update", slow_update)
    async with _client() as client:
        r1, r2 = await asyncio.gather(
            client.post("/api/ingest"), client.post("/api/ingest")
        )
    assert sorted([r1.status_code, r2.status_code]) == [200, 409]
    assert runs == [1]


@pytest.mark.asyncio
async def test_manual_ingest_conflict_when_locked():
    from routers import data as data_router

    async with data_router._INGEST_LOCK:
        async with _client() as client:
            resp = await client.post("/api/ingest")
    assert resp.status_code == 409


# ── Notification reliability gate ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gate_allows_with_few_cycles(monkeypatch):
    import routers.predictions as pred
    import reliability_engine

    async def few_cycles(session, ticker="VOO", window="30d", persist=True):
        return [{"return_percent": -1.0}] * 2

    monkeypatch.setattr(reliability_engine, "generate_trade_cycles", few_cycles)
    allowed, reason = await pred._reliability_gate_allows(None)
    assert allowed is True
    assert "2 cycle" in reason


@pytest.mark.asyncio
async def test_gate_blocks_on_low_win_rate(monkeypatch):
    import routers.predictions as pred
    import reliability_engine

    cycles = [{"return_percent": -1.0}] * 8 + [{"return_percent": 1.0}] * 2

    async def losing(session, ticker="VOO", window="30d", persist=True):
        return cycles

    def metrics(c):
        return {"win_rate": 0.2}

    monkeypatch.setattr(reliability_engine, "generate_trade_cycles", losing)
    monkeypatch.setattr(reliability_engine, "compute_metrics", metrics)
    allowed, reason = await pred._reliability_gate_allows(None)
    assert allowed is False
    assert "below" in reason


@pytest.mark.asyncio
async def test_gate_allows_on_good_win_rate(monkeypatch):
    import reliability_engine
    import routers.predictions as pred

    cycles = [{"return_percent": 1.0}] * 6

    async def winning(session, ticker="VOO", window="30d", persist=True):
        return cycles

    monkeypatch.setattr(reliability_engine, "generate_trade_cycles", winning)
    monkeypatch.setattr(reliability_engine, "compute_metrics", lambda c: {"win_rate": 1.0})
    allowed, _ = await pred._reliability_gate_allows(None)
    assert allowed is True


@pytest.mark.asyncio
async def test_gate_fails_open_on_error(monkeypatch):
    import reliability_engine
    import routers.predictions as pred

    async def boom(session, ticker="VOO", window="30d", persist=True):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(reliability_engine, "generate_trade_cycles", boom)
    allowed, reason = await pred._reliability_gate_allows(None)
    assert allowed is True
    assert "defaulting to allowed" in reason
