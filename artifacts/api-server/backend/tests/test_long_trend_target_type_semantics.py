"""
Tests for drawdown_event and three_state target-type semantics in LongTrendModel.

Covers:
- predict() calibrates raw P(drawdown) before inverting (not the other way around)
- get_baseline_probability() returns 1 - positive_rate for drawdown_event
- get_baseline_probability() returns 0.5 for three_state
- predict() for three_state does NOT apply the binary calibrator
- _target_aware_base_rate() correctness for all three target types
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_xgb(proba_row: list):
    """Return a mock classifier whose predict_proba always returns proba_row."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([proba_row])
    mock.n_features_in_ = 19
    return mock


class _FakeCalibrator:
    """Calibrator that multiplies its input by a known factor for testability."""

    def __init__(self, factor: float = 2.0, cap: float = 0.9):
        self.factor = factor
        self.cap = cap

    def transform(self, raw: float) -> float:
        return min(self.cap, raw * self.factor)


def _bare_model(target_type: str, positive_rate=None):
    """Create a LongTrendModel with _maybe_reload disabled and target type set."""
    import ml.long_trend as lt_mod

    mdl = lt_mod.LongTrendModel.__new__(lt_mod.LongTrendModel)
    mdl.model = None
    mdl._model_feature_count = None
    mdl.calibrator = None
    mdl._model_loaded = True
    mdl._loaded_mtime = None
    mdl._calibrator_mtime = None
    mdl._calibration_report_mtime = None
    mdl.calibration_base_rate = positive_rate
    mdl._baseline_mode = False
    mdl.last_prediction_was_fallback = False
    mdl._promoted_target_type = target_type
    return mdl


# ---------------------------------------------------------------------------
# predict() — drawdown_event calibration order
# ---------------------------------------------------------------------------

class TestDrawdownCalibrationOrder:
    """The calibrator must be applied to raw P(drawdown) THEN inverted."""

    def test_calibrator_applied_before_inversion(self):
        """calibrate(P(drawdown)) then invert — NOT invert then calibrate."""
        import ml.long_trend as lt_mod

        raw_dd = 0.30  # model outputs P(drawdown) = 0.30
        calibrator = _FakeCalibrator(factor=2.0, cap=0.9)
        calibrated_dd = calibrator.transform(raw_dd)  # = 0.60
        expected = 1.0 - calibrated_dd  # = 0.40 (bearish lean)

        mdl = _bare_model("drawdown_event")
        mdl.model = _dummy_xgb([1.0 - raw_dd, raw_dd])
        mdl._model_feature_count = 19
        mdl.calibrator = calibrator

        with patch.object(mdl, "_maybe_reload"):
            result = mdl.predict(np.zeros((1, 19), dtype=np.float32))

        assert abs(result - expected) < 1e-6, (
            f"Expected calibrate-then-invert={expected:.4f}, got {result:.4f}"
        )

    def test_no_calibrator_inverts_raw(self):
        """Without a calibrator, raw P(drawdown) is inverted directly."""
        import ml.long_trend as lt_mod

        raw_dd = 0.25
        mdl = _bare_model("drawdown_event")
        mdl.model = _dummy_xgb([1.0 - raw_dd, raw_dd])
        mdl._model_feature_count = 19
        mdl.calibrator = None

        with patch.object(mdl, "_maybe_reload"):
            result = mdl.predict(np.zeros((1, 19), dtype=np.float32))

        assert abs(result - (1.0 - raw_dd)) < 1e-6, (
            f"Expected {1.0 - raw_dd:.4f}, got {result:.4f}"
        )

    def test_high_drawdown_prob_gives_bearish_score(self):
        """High P(drawdown) should yield ml_confidence < 0.5 (bearish)."""
        import ml.long_trend as lt_mod

        raw_dd = 0.80
        mdl = _bare_model("drawdown_event")
        mdl.model = _dummy_xgb([1.0 - raw_dd, raw_dd])
        mdl._model_feature_count = 19
        mdl.calibrator = None

        with patch.object(mdl, "_maybe_reload"):
            result = mdl.predict(np.zeros((1, 19), dtype=np.float32))

        assert result < 0.5, (
            f"High drawdown probability should yield bearish confidence, got {result:.4f}"
        )

    def test_low_drawdown_prob_gives_bullish_score(self):
        """Low P(drawdown) should yield ml_confidence > 0.5 (bullish)."""
        import ml.long_trend as lt_mod

        raw_dd = 0.05
        mdl = _bare_model("drawdown_event")
        mdl.model = _dummy_xgb([1.0 - raw_dd, raw_dd])
        mdl._model_feature_count = 19
        mdl.calibrator = None

        with patch.object(mdl, "_maybe_reload"):
            result = mdl.predict(np.zeros((1, 19), dtype=np.float32))

        assert result > 0.5, (
            f"Low drawdown probability should yield bullish confidence, got {result:.4f}"
        )


# ---------------------------------------------------------------------------
# predict() — three_state does NOT apply the binary calibrator
# ---------------------------------------------------------------------------

class TestThreeStateNoCalibratorApplied:
    """three_state predict() must not run the binary probability calibrator."""

    def test_calibrator_not_called_for_three_state(self):
        """Even if a calibrator exists, it must not be applied for three_state."""
        import ml.long_trend as lt_mod

        p_off, p_neutral, p_on = 0.1, 0.3, 0.6
        mdl = _bare_model("three_state")
        mdl.model = _dummy_xgb([p_off, p_neutral, p_on])
        mdl._model_feature_count = 19

        spy_cal = MagicMock()
        spy_cal.transform.side_effect = lambda x: x  # identity
        mdl.calibrator = spy_cal

        with patch.object(mdl, "_maybe_reload"):
            result = mdl.predict(np.zeros((1, 19), dtype=np.float32))

        # Calibrator.transform must NOT have been called
        spy_cal.transform.assert_not_called()

        # Score = P(risk-on) + 0.5 * P(neutral)
        expected = p_on + 0.5 * p_neutral
        assert abs(result - expected) < 1e-6, (
            f"Expected {expected:.4f}, got {result:.4f}"
        )

    def test_three_state_score_formula(self):
        """Collapsed score = P(risk-on) + 0.5 × P(neutral)."""
        import ml.long_trend as lt_mod

        p_off, p_neutral, p_on = 0.5, 0.3, 0.2
        mdl = _bare_model("three_state")
        mdl.model = _dummy_xgb([p_off, p_neutral, p_on])
        mdl._model_feature_count = 19
        mdl.calibrator = None

        with patch.object(mdl, "_maybe_reload"):
            result = mdl.predict(np.zeros((1, 19), dtype=np.float32))

        expected = p_on + 0.5 * p_neutral  # 0.2 + 0.15 = 0.35 (bearish)
        assert abs(result - expected) < 1e-6, (
            f"Expected {expected:.4f}, got {result:.4f}"
        )
        assert result < 0.5, "Risk-off dominant should yield bearish confidence"

    def test_three_state_risk_on_dominant_bullish(self):
        """Risk-on dominant three_state should yield ml_confidence > 0.5."""
        import ml.long_trend as lt_mod

        p_off, p_neutral, p_on = 0.1, 0.1, 0.8
        mdl = _bare_model("three_state")
        mdl.model = _dummy_xgb([p_off, p_neutral, p_on])
        mdl._model_feature_count = 19
        mdl.calibrator = None

        with patch.object(mdl, "_maybe_reload"):
            result = mdl.predict(np.zeros((1, 19), dtype=np.float32))

        assert result > 0.5, f"Risk-on dominant should be bullish, got {result:.4f}"


# ---------------------------------------------------------------------------
# _target_aware_base_rate() and get_baseline_probability()
# ---------------------------------------------------------------------------

class TestTargetAwareBaseRate:
    """_target_aware_base_rate adjusts positive_rate by target semantics."""

    def test_direction_returns_positive_rate(self):
        """For direction models the raw positive_rate is returned unchanged."""
        import ml.long_trend as lt_mod

        mdl = _bare_model("direction", positive_rate=0.73)
        rate = mdl._target_aware_base_rate()
        assert abs(rate - 0.73) < 1e-6, f"Expected 0.73, got {rate}"

    def test_drawdown_returns_one_minus_positive_rate(self):
        """For drawdown_event, baseline = 1 - positive_rate (inverted semantics)."""
        import ml.long_trend as lt_mod

        # calibration report positive_rate = 0.08 (8% event prevalence)
        mdl = _bare_model("drawdown_event", positive_rate=0.08)
        rate = mdl._target_aware_base_rate()
        assert abs(rate - 0.92) < 1e-6, f"Expected 0.92 (1-0.08), got {rate}"

    def test_three_state_returns_half(self):
        """For three_state, baseline is neutral 0.5 regardless of positive_rate."""
        import ml.long_trend as lt_mod

        mdl = _bare_model("three_state", positive_rate=0.35)
        rate = mdl._target_aware_base_rate()
        assert rate == 0.5, f"Expected 0.5 for three_state, got {rate}"

    def test_none_when_no_rate_loaded(self):
        """Returns None when calibration_base_rate has not been loaded."""
        import ml.long_trend as lt_mod

        mdl = _bare_model("direction", positive_rate=None)
        assert mdl._target_aware_base_rate() is None

    def test_drawdown_boundary_clamped(self):
        """Inverted rate is clamped to (0.01, 0.99)."""
        import ml.long_trend as lt_mod

        # positive_rate near 0 → inverted value near 1 → capped at 0.99
        mdl = _bare_model("drawdown_event", positive_rate=0.001)
        rate = mdl._target_aware_base_rate()
        assert rate <= 0.99, f"Expected <= 0.99, got {rate}"

    def test_get_baseline_probability_drawdown(self):
        """get_baseline_probability() returns 1 - positive_rate for drawdown_event."""
        import ml.long_trend as lt_mod

        mdl = _bare_model("drawdown_event", positive_rate=0.05)
        with patch.object(mdl, "_maybe_reload"):
            baseline = mdl.get_baseline_probability()
        assert abs(baseline - 0.95) < 1e-6, f"Expected 0.95, got {baseline}"

    def test_get_baseline_probability_three_state(self):
        """get_baseline_probability() returns 0.5 for three_state."""
        import ml.long_trend as lt_mod

        mdl = _bare_model("three_state", positive_rate=0.40)
        with patch.object(mdl, "_maybe_reload"):
            baseline = mdl.get_baseline_probability()
        assert baseline == 0.5, f"Expected 0.5 for three_state, got {baseline}"

    def test_get_baseline_probability_direction(self):
        """get_baseline_probability() returns positive_rate for direction."""
        import ml.long_trend as lt_mod

        mdl = _bare_model("direction", positive_rate=0.73)
        with patch.object(mdl, "_maybe_reload"):
            baseline = mdl.get_baseline_probability()
        assert abs(baseline - 0.73) < 1e-6, f"Expected 0.73, got {baseline}"

    def test_get_baseline_probability_fallback_when_no_rate(self):
        """Falls back to 0.5 when no calibration rate has been loaded."""
        import ml.long_trend as lt_mod

        mdl = _bare_model("drawdown_event", positive_rate=None)
        with patch.object(mdl, "_maybe_reload"):
            baseline = mdl.get_baseline_probability()
        assert baseline == 0.5, f"Expected 0.5 fallback, got {baseline}"
