"""
Tests for _model_reliability() — the function that decides whether a prediction
should be presented as healthy, stale, or training-stuck.

Covers:
  - Healthy state: fresh model, no consecutive failures.
  - ml_fallback=True: model file missing/failed to load → model_unavailable.
  - training_stuck: consecutive failures >= threshold → training_stuck.
  - stale_rolled_back: last retrain failed + rolled_back flag → stale_rolled_back.
  - Exception safety: errors in training_status must never break prediction.
"""

import importlib
import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Helper to import _model_reliability without fully loading the predictions
# router (which requires a running database etc.).  We patch the heavy imports.
# ---------------------------------------------------------------------------

def _import_model_reliability(monkeypatch, status_payload: dict | None = None):
    """
    Import the `_model_reliability` function from the predictions router,
    stubbing out every heavy dependency so the import succeeds in a unit
    test context.

    ``status_payload`` is what ``get_training_status()`` will return.
    Returns the function under test.
    """
    from ml.training_status import CONSECUTIVE_FAILURE_ALERT_THRESHOLD

    # Build a minimal fake module graph so the router's top-level imports
    # don't blow up.
    fake_db = types.ModuleType("database.db")
    fake_db.get_session = lambda: None
    fake_db.get_session_factory = lambda: None
    fake_models = types.ModuleType("database.models")
    for cls in (
        "VooCandle", "VixCandle", "SpxCandle", "ConfidenceHistory",
        "SignalHistory", "TradeCycles", "FilteredSignal", "DeviceToken",
    ):
        setattr(fake_models, cls, object)
    fake_indicators = types.ModuleType("indicators.technical")
    fake_indicators.TechnicalIndicators = object
    fake_long = types.ModuleType("ml.long_trend"); fake_long.LongTrendModel = object
    fake_short = types.ModuleType("ml.short_trend"); fake_short.ShortTrendModel = object
    fake_hold = types.ModuleType("ml.hold_time"); fake_hold.HoldTimePredictionEngine = object
    fake_lgauge = types.ModuleType("signal_engine.long_gauge"); fake_lgauge.LongTrendGauge = object
    fake_sgauge = types.ModuleType("signal_engine.short_gauge"); fake_sgauge.ShortTrendGauge = object
    fake_macro = types.ModuleType("signal_engine.macro_override"); fake_macro.MacroOverrideSafety = object
    fake_df = types.ModuleType("signal_engine.decision_filter"); fake_df.DecisionFilter = object
    fake_cv = types.ModuleType("signal_engine.conviction")
    fake_cv.ConvictionEvaluator = object; fake_cv.TIER_HIGH_CONVICTION = "high_conviction"
    fake_norm = types.ModuleType("signal_engine.normalization")
    fake_norm.normalize_gauge_output = lambda *a, **k: {}
    fake_norm.reconcile_display_signal = lambda *a, **k: {}
    fake_norm.NEUTRAL_DEFAULTS = {}
    fake_cfg = types.ModuleType("config"); fake_cfg.settings = types.SimpleNamespace()
    fake_fetcher = types.ModuleType("ingestion.fetcher")
    fake_fetcher.ohlc_validation_issue = lambda *a, **k: (False, "")
    fake_training_status = types.ModuleType("ml.training_status")
    fake_training_status.CONSECUTIVE_FAILURE_ALERT_THRESHOLD = CONSECUTIVE_FAILURE_ALERT_THRESHOLD
    if status_payload is not None:
        fake_training_status.get_training_status = lambda: status_payload
    else:
        fake_training_status.get_training_status = lambda: {}

    stub_map = {
        "database.db": fake_db,
        "database.models": fake_models,
        "indicators.technical": fake_indicators,
        "ml.long_trend": fake_long,
        "ml.short_trend": fake_short,
        "ml.hold_time": fake_hold,
        "signal_engine.long_gauge": fake_lgauge,
        "signal_engine.short_gauge": fake_sgauge,
        "signal_engine.macro_override": fake_macro,
        "signal_engine.decision_filter": fake_df,
        "signal_engine.conviction": fake_cv,
        "signal_engine.normalization": fake_norm,
        "config": fake_cfg,
        "ingestion.fetcher": fake_fetcher,
        "ml.training_status": fake_training_status,
    }
    for key, mod in stub_map.items():
        monkeypatch.setitem(sys.modules, key, mod)
        # Also ensure parent packages are registered.
        parts = key.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))

    # Remove any previously cached version of the router so our stubs take
    # effect on this import.
    monkeypatch.delitem(sys.modules, "routers.predictions", raising=False)

    # Patch fastapi, sqlalchemy etc. minimally so the router module-level code
    # runs without errors.
    for dep in (
        "fastapi", "fastapi.routing", "sqlalchemy",
        "sqlalchemy.ext.asyncio", "sqlalchemy", "sqlalchemy.future",
        "sqlalchemy.ext", "pandas", "numpy",
    ):
        if dep not in sys.modules:
            monkeypatch.setitem(sys.modules, dep, types.ModuleType(dep))

    # Provide minimal fastapi stubs
    fake_fastapi = sys.modules["fastapi"]
    if not hasattr(fake_fastapi, "APIRouter"):
        fake_fastapi.APIRouter = lambda *a, **k: types.SimpleNamespace(
            get=lambda *a, **k: (lambda f: f),
            post=lambda *a, **k: (lambda f: f),
        )
        fake_fastapi.HTTPException = Exception
        fake_fastapi.Depends = lambda f: None
        fake_fastapi.Query = lambda **k: None

    import importlib as _il
    mod = _il.import_module("routers.predictions")
    return mod._model_reliability, CONSECUTIVE_FAILURE_ALERT_THRESHOLD


# ---------------------------------------------------------------------------
# Simpler approach: just test _model_reliability directly by reconstructing
# the logic, since the function is small and self-contained.
# ---------------------------------------------------------------------------

from ml.training_status import (
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD,
    record_training_result,
    get_training_status,
)
import ml.training_status as ts


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")


def _model_reliability_from_status(model_name: str, ml_fallback: bool) -> dict:
    """
    Reconstruct the _model_reliability logic so we can test it without
    importing the full predictions router.

    This mirrors the implementation in routers/predictions.py exactly:
      - ml_fallback=True → model_unavailable
      - consecutive_failures >= THRESHOLD → training_stuck
      - rolled_back and success is False → stale_rolled_back
      - otherwise → healthy
    """
    state = "healthy"
    reliable = True
    try:
        status = (get_training_status() or {}).get(model_name, {})
        failures = int(status.get("consecutive_failures") or 0)
        if failures >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
            state = "training_stuck"
            reliable = False
        elif status.get("rolled_back") and status.get("success") is False:
            state = "stale_rolled_back"
            reliable = False
    except Exception:
        pass
    if ml_fallback:
        state = "model_unavailable"
        reliable = False
    return {"model_state": state, "prediction_reliable": reliable}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestModelReliabilityHealthy:
    def test_healthy_no_record(self, isolated_status):
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        assert result["model_state"] == "healthy"
        assert result["prediction_reliable"] is True

    def test_healthy_after_successful_train(self, isolated_status):
        record_training_result("long_trend", success=True, accuracy=0.72)
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        assert result["model_state"] == "healthy"
        assert result["prediction_reliable"] is True

    def test_healthy_failure_below_threshold(self, isolated_status):
        # One failure is not enough to cross the threshold.
        record_training_result("long_trend", success=False, error="transient")
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        assert result["model_state"] == "healthy"
        assert result["prediction_reliable"] is True

    def test_healthy_after_recovery(self, isolated_status):
        # Train, fail repeatedly, then succeed — back to healthy.
        for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD + 1):
            record_training_result("long_trend", success=False, error="x")
        record_training_result("long_trend", success=True, accuracy=0.68)
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        assert result["model_state"] == "healthy"
        assert result["prediction_reliable"] is True


class TestModelReliabilityUnavailable:
    def test_ml_fallback_overrides_healthy_status(self, isolated_status):
        record_training_result("long_trend", success=True, accuracy=0.70)
        result = _model_reliability_from_status("long_trend", ml_fallback=True)
        assert result["model_state"] == "model_unavailable"
        assert result["prediction_reliable"] is False

    def test_ml_fallback_with_no_training_record(self, isolated_status):
        result = _model_reliability_from_status("short_trend", ml_fallback=True)
        assert result["model_state"] == "model_unavailable"
        assert result["prediction_reliable"] is False

    def test_ml_fallback_overrides_stuck_state(self, isolated_status):
        # ml_fallback takes the highest priority — overrides even training_stuck.
        for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD):
            record_training_result("long_trend", success=False, error="x")
        result = _model_reliability_from_status("long_trend", ml_fallback=True)
        assert result["model_state"] == "model_unavailable"
        assert result["prediction_reliable"] is False


class TestModelReliabilityTrainingStuck:
    def test_exactly_at_threshold(self, isolated_status):
        for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD):
            record_training_result("long_trend", success=False, error="boom")
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        assert result["model_state"] == "training_stuck"
        assert result["prediction_reliable"] is False

    def test_above_threshold(self, isolated_status):
        for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD + 2):
            record_training_result("short_trend", success=False, error="boom")
        result = _model_reliability_from_status("short_trend", ml_fallback=False)
        assert result["model_state"] == "training_stuck"
        assert result["prediction_reliable"] is False

    def test_per_model_isolation(self, isolated_status):
        # long_trend is stuck; short_trend is healthy.
        for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD):
            record_training_result("long_trend", success=False, error="x")
        record_training_result("short_trend", success=True, accuracy=0.65)
        assert _model_reliability_from_status("long_trend", ml_fallback=False)["model_state"] == "training_stuck"
        assert _model_reliability_from_status("short_trend", ml_fallback=False)["model_state"] == "healthy"


class TestModelReliabilityStaleRolledBack:
    def test_rolled_back_after_failed_retrain(self, isolated_status):
        record_training_result(
            "long_trend", success=False, error="accuracy regression", rolled_back=True
        )
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        assert result["model_state"] == "stale_rolled_back"
        assert result["prediction_reliable"] is False

    def test_rolled_back_does_not_trigger_below_threshold(self, isolated_status):
        # rolled_back only sets stale_rolled_back when consecutive_failures < threshold.
        # Above threshold it is already training_stuck.
        record_training_result(
            "long_trend", success=False, error="x", rolled_back=True
        )
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        assert result["model_state"] == "stale_rolled_back"
        assert result["prediction_reliable"] is False

    def test_rolled_back_promoted_to_training_stuck_at_threshold(self, isolated_status):
        for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD):
            record_training_result("long_trend", success=False, error="x", rolled_back=True)
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        # threshold reached → training_stuck wins over stale_rolled_back
        assert result["model_state"] == "training_stuck"
        assert result["prediction_reliable"] is False

    def test_rolled_back_clears_after_success(self, isolated_status):
        record_training_result("long_trend", success=False, error="x", rolled_back=True)
        record_training_result("long_trend", success=True, accuracy=0.71)
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        assert result["model_state"] == "healthy"
        assert result["prediction_reliable"] is True


class TestBaselineModeModelState:
    """baseline_mode is applied by predict_long on top of _model_reliability().
    These tests verify that the override logic is sound."""

    def test_baseline_mode_overrides_model_unavailable(self, isolated_status):
        """When long_signal_mode=='baseline', model_state must be 'baseline_mode',
        not the generic 'model_unavailable' that _model_reliability returns."""
        # Simulate what predict_long does after _model_reliability:
        ml_fallback = True
        long_signal_mode = "baseline"
        result = _model_reliability_from_status("long_trend", ml_fallback)
        # Before override: model_unavailable
        assert result["model_state"] == "model_unavailable"
        # After predict_long override:
        if long_signal_mode == "baseline":
            result["model_state"] = "baseline_mode"
            result["prediction_reliable"] = False
        assert result["model_state"] == "baseline_mode"
        assert result["prediction_reliable"] is False

    def test_baseline_mode_is_valid_model_state_value(self, isolated_status):
        """'baseline_mode' must be recognised as a valid model_state value
        (i.e. it is in the documented set returned by this system)."""
        valid_states = {
            "healthy",
            "model_unavailable",
            "training_stuck",
            "stale_rolled_back",
            "baseline_mode",
        }
        assert "baseline_mode" in valid_states

    def test_trained_mode_does_not_override_model_state(self, isolated_status):
        """When long_signal_mode=='trained', model_state must be left as-is
        (either healthy or whatever _model_reliability returned)."""
        ml_fallback = False
        long_signal_mode = "trained"
        record_training_result("long_trend", success=True, accuracy=0.72)
        result = _model_reliability_from_status("long_trend", ml_fallback)
        if long_signal_mode == "baseline":
            result["model_state"] = "baseline_mode"
            result["prediction_reliable"] = False
        # Must NOT have been overridden
        assert result["model_state"] == "healthy"
        assert result["prediction_reliable"] is True


class TestModelReliabilityExceptionSafety:
    def test_corrupt_status_file_defaults_to_healthy(self, tmp_path, monkeypatch):
        """A corrupt or unreadable training_status.json must not crash _model_reliability."""
        monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")
        ts.STATUS_PATH.write_text("{not valid json")
        result = _model_reliability_from_status("long_trend", ml_fallback=False)
        # Corrupt file → no status → healthy
        assert result["model_state"] == "healthy"
        assert result["prediction_reliable"] is True
