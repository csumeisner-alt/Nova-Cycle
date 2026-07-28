"""
Tests for signal_engine.normalization (Task: normalized gauge confidence)
and the new confidence_percent/trend/display_signal fields on the
prediction endpoints.
"""

import math

import pytest

from signal_engine.normalization import (
    normalize_gauge_output,
    reconcile_display_signal,
    NEUTRAL_DEFAULTS,
    SIGMOID_SCALE,
    NEUTRAL_BAND,
    SIGNAL_CONFIDENCE_THRESHOLD,
    TREND_UP, TREND_DOWN, TREND_NEUTRAL,
    SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD,
)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestNormalizeGaugeOutput:

    def test_zero_score_is_neutral(self):
        out = normalize_gauge_output(0.0)
        assert out == {"confidence_percent": 0,
                       "trend": TREND_NEUTRAL,
                       "display_signal": SIGNAL_HOLD}

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"),
                                     float("-inf"), "abc", {}, []])
    def test_invalid_inputs_return_neutral_defaults(self, bad):
        assert normalize_gauge_output(bad) == NEUTRAL_DEFAULTS

    @pytest.mark.parametrize("score", [-100, -70, -5, 0, 5, 70, 100,
                                       1e9, -1e9, 12345.678])
    def test_confidence_percent_always_in_range(self, score):
        cp = normalize_gauge_output(score)["confidence_percent"]
        assert isinstance(cp, int)
        assert 0 <= cp <= 100

    def test_extreme_scores_clamp_to_100_not_beyond(self):
        assert normalize_gauge_output(1e6)["confidence_percent"] == 100
        assert normalize_gauge_output(-1e6)["confidence_percent"] == 100

    def test_symmetry_positive_negative(self):
        pos = normalize_gauge_output(42.0)
        neg = normalize_gauge_output(-42.0)
        assert pos["confidence_percent"] == neg["confidence_percent"]
        assert pos["trend"] == TREND_UP
        assert neg["trend"] == TREND_DOWN

    def test_monotonic_in_magnitude(self):
        values = [normalize_gauge_output(s)["confidence_percent"]
                  for s in (0, 10, 25, 50, 75, 100)]
        assert values == sorted(values)

    # ── Trend neutral band ──────────────────────────────────────────────
    def test_neutral_band_edges(self):
        assert normalize_gauge_output(NEUTRAL_BAND)["trend"] == TREND_NEUTRAL
        assert normalize_gauge_output(-NEUTRAL_BAND)["trend"] == TREND_NEUTRAL
        assert normalize_gauge_output(NEUTRAL_BAND + 0.01)["trend"] == TREND_UP
        assert normalize_gauge_output(-NEUTRAL_BAND - 0.01)["trend"] == TREND_DOWN

    # ── 65% display-signal threshold edges ──────────────────────────────
    def _score_for_percent(self, pct: float) -> float:
        """Invert the normalization to find the raw score giving `pct`%."""
        normalized = pct / 100.0
        sig = normalized / 2.0 + 0.5
        return SIGMOID_SCALE * math.log(sig / (1.0 - sig))

    def test_signal_threshold_edge_65(self):
        s65 = self._score_for_percent(65.0)
        out = normalize_gauge_output(s65 + 0.5)
        assert out["confidence_percent"] >= SIGNAL_CONFIDENCE_THRESHOLD
        assert out["trend"] == TREND_UP
        assert out["display_signal"] == SIGNAL_BUY

        out_sell = normalize_gauge_output(-(s65 + 0.5))
        assert out_sell["display_signal"] == SIGNAL_SELL

    def test_signal_threshold_edge_64_is_hold(self):
        s64 = self._score_for_percent(64.0)
        out = normalize_gauge_output(s64 - 0.5)
        assert out["confidence_percent"] < SIGNAL_CONFIDENCE_THRESHOLD
        assert out["display_signal"] == SIGNAL_HOLD

    def test_high_confidence_but_neutral_trend_is_hold(self):
        # Trend gate: even a high percent with NEUTRAL trend must be HOLD.
        # (Can only occur if NEUTRAL_BAND produced a low percent — assert the
        # invariant explicitly.)
        out = normalize_gauge_output(NEUTRAL_BAND)
        assert out["display_signal"] == SIGNAL_HOLD

    def test_max_scores(self):
        out = normalize_gauge_output(100.0)
        assert out["trend"] == TREND_UP
        assert out["display_signal"] == SIGNAL_BUY
        out = normalize_gauge_output(-100.0)
        assert out["trend"] == TREND_DOWN
        assert out["display_signal"] == SIGNAL_SELL


# ---------------------------------------------------------------------------
# reconcile_display_signal: display bias must never contradict an
# override-suppressed filtered signal.
# ---------------------------------------------------------------------------

class TestReconcileDisplaySignal:

    def _buy_bias(self):
        out = normalize_gauge_output(100.0)
        assert out["display_signal"] == SIGNAL_BUY
        return out

    def test_override_neutral_downgrades_buy_bias_to_hold(self):
        out = reconcile_display_signal(self._buy_bias(), "neutral", True)
        assert out["display_signal"] == SIGNAL_HOLD

    def test_override_neutral_downgrades_sell_bias_to_hold(self):
        sell = normalize_gauge_output(-100.0)
        out = reconcile_display_signal(sell, "neutral", True)
        assert out["display_signal"] == SIGNAL_HOLD

    def test_trend_and_confidence_untouched(self):
        buy = self._buy_bias()
        out = reconcile_display_signal(buy, "neutral", True)
        assert out["trend"] == buy["trend"]
        assert out["confidence_percent"] == buy["confidence_percent"]

    def test_no_override_keeps_bias(self):
        out = reconcile_display_signal(self._buy_bias(), "buy", False)
        assert out["display_signal"] == SIGNAL_BUY

    def test_override_but_signal_not_neutral_keeps_bias(self):
        # e.g. decision filter later changed the signal — only the
        # override-forced-neutral case downgrades.
        out = reconcile_display_signal(self._buy_bias(), "buy", True)
        assert out["display_signal"] == SIGNAL_BUY

    def test_does_not_mutate_input(self):
        buy = self._buy_bias()
        reconcile_display_signal(buy, "neutral", True)
        assert buy["display_signal"] == SIGNAL_BUY

    def test_invalid_normalized_returns_neutral_defaults(self):
        assert reconcile_display_signal(None, "neutral", True) == NEUTRAL_DEFAULTS


# ---------------------------------------------------------------------------
# Endpoint tests: new fields always present and in range
# ---------------------------------------------------------------------------

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.db import get_session
from database.models import Base
from main import app
from tests.test_prediction_endpoints import (
    _daily_candles, _fivemin_candles, _vix_candles,
)

VALID_TRENDS = {TREND_UP, TREND_DOWN, TREND_NEUTRAL}
VALID_SIGNALS = {SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD}


def _assert_normalized_fields(body: dict):
    assert "confidence_percent" in body
    assert "trend" in body
    assert "display_signal" in body
    assert isinstance(body["confidence_percent"], int)
    assert 0 <= body["confidence_percent"] <= 100
    assert body["trend"] in VALID_TRENDS
    assert body["display_signal"] in VALID_SIGNALS
    # Legacy fields must remain for installed clients.
    for legacy in ("score", "signal", "confidence"):
        assert legacy in body


async def _make_client(tmp_path, seed: bool):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'norm_test.db'}",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    if seed:
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
    return engine, AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest_asyncio.fixture
async def seeded_client(tmp_path):
    engine, ac = await _make_client(tmp_path, seed=True)
    try:
        async with ac as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_session, None)
        await engine.dispose()


@pytest_asyncio.fixture
async def empty_client(tmp_path):
    engine, ac = await _make_client(tmp_path, seed=False)
    try:
        async with ac as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_session, None)
        await engine.dispose()


@pytest.mark.parametrize("endpoint", ["/api/predict_long", "/api/predict_short"])
async def test_prediction_endpoints_include_normalized_fields(seeded_client, endpoint):
    resp = await seeded_client.post(endpoint, params={"ticker": "VOO"})
    assert resp.status_code == 200
    _assert_normalized_fields(resp.json())


async def test_predict_short_override_forces_hold_display(seeded_client, monkeypatch):
    """When the macro override suppresses the signal, display_signal must be
    HOLD even if the raw gauge score would produce BUY BIAS."""
    from routers import predictions as preds

    monkeypatch.setattr(
        preds._short_gauge, "compute_score",
        lambda *a, **k: {"score": 100.0, "signal": "buy",
                         "confidence": 0.9, "breakdown": {}},
    )
    monkeypatch.setattr(
        preds._macro_override, "apply_override",
        lambda *a, **k: {"override_applied": True, "reason": "test override"},
    )

    resp = await seeded_client.post("/api/predict_short", params={"ticker": "VOO"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["macro_override_applied"] is True
    assert body["signal"] == "neutral"
    assert body["display_signal"] == SIGNAL_HOLD
    # Factual gauge readings remain untouched.
    assert body["trend"] == TREND_UP
    assert body["confidence_percent"] >= SIGNAL_CONFIDENCE_THRESHOLD


@pytest.mark.parametrize("endpoint", ["/api/predict_long", "/api/predict_short"])
async def test_prediction_endpoints_no_data_returns_neutral(empty_client, endpoint):
    resp = await empty_client.post(endpoint, params={"ticker": "VOO"})
    assert resp.status_code == 200
    body = resp.json()
    _assert_normalized_fields(body)
    assert body["confidence_percent"] == 0
    assert body["trend"] == TREND_NEUTRAL
    assert body["display_signal"] == SIGNAL_HOLD
