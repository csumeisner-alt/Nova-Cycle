"""
Regression tests for the training-stuck notification gate.

The prediction endpoints suppress push notifications when the model is
training-stuck (`prediction_reliable=False`), even when the gauge emits an
actionable buy/sell signal.  These tests exercise the *real* endpoint code
path end-to-end (real FastAPI app, seeded SQLite DB, real training_status
module writing to an isolated tmp file) and verify:

  1. With consecutive_failures >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD,
     _notify_all_devices_bg is NEVER called from predict_long / predict_short,
     even for a forced buy signal.
  2. Once training recovers (a successful retrain resets the failure count),
     notifications fire again.
"""

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.db import get_session
from database.models import Base
from main import app

import ml.training_status as ts
from ml.training_status import (
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD,
    record_training_result,
)

import routers.predictions as preds

from tests.test_prediction_endpoints import (
    _daily_candles,
    _fivemin_candles,
    _vix_candles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(tmp_path):
    """Real app + isolated seeded SQLite DB (same pattern as the e2e tests)."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'gate_test.db'}",
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


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    """Point the real training_status module at an isolated tmp JSON file."""
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")


@pytest.fixture
def notify_recorder(monkeypatch):
    """Replace _notify_all_devices_bg with a synchronous-recording stub.

    The call kwargs are recorded at coroutine-creation time (i.e. exactly when
    the endpoint decides to notify), so the assertion does not depend on the
    background task ever being scheduled or completing.
    """
    calls = []

    def _fake_notify(**kwargs):
        calls.append(kwargs)

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(preds, "_notify_all_devices_bg", _fake_notify)
    return calls


@pytest.fixture
def force_buy_signal(monkeypatch):
    """Force the decision filter to emit an executable buy signal so the
    notification branch is reached regardless of what the seeded market data
    happens to produce."""

    def _evaluate(**kwargs):
        return {
            "final_signal": "buy",
            "is_candidate": False,
            "candidate_signal": None,
            "priority_boost": 0.0,
            "decision_penalty": 0.0,
            "volatility_regime": "calm",
            "cycle_quality_score": 0.8,
            "conviction_tier_cap": None,
            "reason": "forced buy for notification-gate test",
        }

    monkeypatch.setattr(preds._decision_filter, "evaluate", _evaluate)


def _force_training_stuck(model_name: str):
    for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD):
        record_training_result(model_name, success=False, error="retrain boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNotificationsSuppressedWhileTrainingStuck:
    async def test_predict_long_buy_signal_does_not_notify(
        self, client, isolated_status, notify_recorder, force_buy_signal
    ):
        _force_training_stuck("long_trend")

        resp = await client.post("/api/predict_long", params={"ticker": "VOO"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["signal"] == "buy"
        assert body["model_state"] == "training_stuck"
        assert body["prediction_reliable"] is False
        assert notify_recorder == [], (
            f"notification sent while long model training-stuck: {notify_recorder}"
        )

    async def test_predict_short_buy_signal_does_not_notify(
        self, client, isolated_status, notify_recorder, force_buy_signal
    ):
        _force_training_stuck("short_trend")

        resp = await client.post("/api/predict_short", params={"ticker": "VOO"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["signal"] == "buy"
        assert body["model_state"] == "training_stuck"
        assert body["prediction_reliable"] is False
        assert notify_recorder == [], (
            f"notification sent while short model training-stuck: {notify_recorder}"
        )

    async def test_stuck_models_are_gated_independently(
        self, client, isolated_status, notify_recorder, force_buy_signal
    ):
        """A stuck long model must not suppress a healthy short model."""
        _force_training_stuck("long_trend")
        record_training_result("short_trend", success=True, accuracy=0.66)

        resp = await client.post("/api/predict_long", params={"ticker": "VOO"})
        assert resp.json()["prediction_reliable"] is False
        assert notify_recorder == []

        resp = await client.post("/api/predict_short", params={"ticker": "VOO"})
        assert resp.json()["prediction_reliable"] is True
        assert len(notify_recorder) == 1
        assert notify_recorder[0]["gauge_type"] == "short"


class TestNotificationsResumeAfterRecovery:
    async def test_predict_long_notifies_after_recovery(
        self, client, isolated_status, notify_recorder, force_buy_signal
    ):
        _force_training_stuck("long_trend")
        # While stuck: no notification.
        await client.post("/api/predict_long", params={"ticker": "VOO"})
        assert notify_recorder == []

        # Recovery: a successful retrain resets consecutive_failures.
        record_training_result("long_trend", success=True, accuracy=0.71)

        resp = await client.post("/api/predict_long", params={"ticker": "VOO"})
        body = resp.json()
        assert body["signal"] == "buy"
        assert body["model_state"] == "healthy"
        assert body["prediction_reliable"] is True
        assert len(notify_recorder) == 1
        assert notify_recorder[0]["gauge_type"] == "long"
        assert notify_recorder[0]["signal_type"] == "buy"

    async def test_predict_short_notifies_after_recovery(
        self, client, isolated_status, notify_recorder, force_buy_signal
    ):
        _force_training_stuck("short_trend")
        await client.post("/api/predict_short", params={"ticker": "VOO"})
        assert notify_recorder == []

        record_training_result("short_trend", success=True, accuracy=0.63)

        resp = await client.post("/api/predict_short", params={"ticker": "VOO"})
        body = resp.json()
        assert body["signal"] == "buy"
        assert body["model_state"] == "healthy"
        assert body["prediction_reliable"] is True
        assert len(notify_recorder) == 1
        assert notify_recorder[0]["gauge_type"] == "short"
        assert notify_recorder[0]["signal_type"] == "buy"
