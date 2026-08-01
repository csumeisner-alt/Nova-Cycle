"""
MacroOverrideSafety boundary tests
====================================
Verifies that the suppression thresholds (LONG_STRONG_BULL / LONG_STRONG_BEAR)
are aligned with the long-gauge signal thresholds (LONG_BUY_THRESHOLD /
LONG_SELL_THRESHOLD) and fire at the correct score boundaries.

Key regression guarded here
---------------------------
Previously the constants were hardcoded to ±70 while BUY/SELL signals fire at
±65.  That left a 65–70 band where a confirmed long-trend signal existed but
macro suppression never engaged.  These tests ensure the gap is closed and stays
closed even if the threshold values change in config.py.
"""

from __future__ import annotations

import pytest

from config import settings
from signal_engine.macro_override import (
    LONG_STRONG_BEAR,
    LONG_STRONG_BULL,
    ML_OVERRIDE_THRESHOLD,
    MacroOverrideSafety,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOW_ML = ML_OVERRIDE_THRESHOLD - 0.01   # just below → suppression allowed
HIGH_ML = ML_OVERRIDE_THRESHOLD + 0.01  # just above → ML bypass kicks in


@pytest.fixture
def safety() -> MacroOverrideSafety:
    return MacroOverrideSafety()


# ---------------------------------------------------------------------------
# 1. Constant alignment: module-level constants must equal config thresholds
# ---------------------------------------------------------------------------

class TestConstantAlignment:
    """LONG_STRONG_BULL/BEAR must track settings.LONG_BUY/SELL_THRESHOLD."""

    def test_bull_constant_equals_long_buy_threshold(self):
        assert LONG_STRONG_BULL == settings.LONG_BUY_THRESHOLD, (
            f"LONG_STRONG_BULL ({LONG_STRONG_BULL}) != "
            f"settings.LONG_BUY_THRESHOLD ({settings.LONG_BUY_THRESHOLD}). "
            "Update macro_override.py to stay in sync with config."
        )

    def test_bear_constant_equals_long_sell_threshold(self):
        assert LONG_STRONG_BEAR == settings.LONG_SELL_THRESHOLD, (
            f"LONG_STRONG_BEAR ({LONG_STRONG_BEAR}) != "
            f"settings.LONG_SELL_THRESHOLD ({settings.LONG_SELL_THRESHOLD}). "
            "Update macro_override.py to stay in sync with config."
        )

    def test_bull_is_positive_bear_is_negative(self):
        assert LONG_STRONG_BULL > 0
        assert LONG_STRONG_BEAR < 0

    def test_bull_and_bear_are_symmetric(self):
        assert LONG_STRONG_BULL == -LONG_STRONG_BEAR, (
            "Thresholds should be symmetric; check config.py."
        )


# ---------------------------------------------------------------------------
# 2. Bearish suppression: long_score < LONG_STRONG_BEAR suppresses BUY
# ---------------------------------------------------------------------------

class TestBearishSuppression:
    """strong bearish trend (score < LONG_STRONG_BEAR) → short BUY suppressed."""

    def test_just_below_bear_threshold_suppresses_buy(self, safety):
        score = LONG_STRONG_BEAR - 0.01   # e.g. -65.01
        result = safety.apply_override(score, "buy", LOW_ML)
        assert result["allowed"] is False
        assert result["override_applied"] is True

    def test_at_bear_threshold_does_not_suppress(self, safety):
        # Boundary is strict (<), so exactly at the threshold should NOT suppress.
        score = LONG_STRONG_BEAR              # e.g. -65.0
        result = safety.apply_override(score, "buy", LOW_ML)
        assert result["allowed"] is True
        assert result["override_applied"] is False

    def test_well_below_bear_threshold_suppresses_buy(self, safety):
        score = LONG_STRONG_BEAR - 10.0      # e.g. -75.0
        result = safety.apply_override(score, "buy", LOW_ML)
        assert result["allowed"] is False
        assert result["override_applied"] is True

    def test_bearish_buy_suppression_with_high_ml_bypassed(self, safety):
        """High ML confidence overrides bearish suppression."""
        score = LONG_STRONG_BEAR - 5.0
        result = safety.apply_override(score, "buy", HIGH_ML)
        assert result["allowed"] is True
        assert result["override_applied"] is False

    def test_bearish_does_not_suppress_sell(self, safety):
        """Bearish long trend + short SELL are aligned; no suppression."""
        score = LONG_STRONG_BEAR - 5.0
        result = safety.apply_override(score, "sell", LOW_ML)
        assert result["allowed"] is True
        assert result["override_applied"] is False


# ---------------------------------------------------------------------------
# 3. Bullish suppression: long_score > LONG_STRONG_BULL suppresses SELL
# ---------------------------------------------------------------------------

class TestBullishSuppression:
    """strong bullish trend (score > LONG_STRONG_BULL) → short SELL suppressed."""

    def test_just_above_bull_threshold_suppresses_sell(self, safety):
        score = LONG_STRONG_BULL + 0.01      # e.g. +65.01
        result = safety.apply_override(score, "sell", LOW_ML)
        assert result["allowed"] is False
        assert result["override_applied"] is True

    def test_at_bull_threshold_does_not_suppress(self, safety):
        # Boundary is strict (>), so exactly at the threshold should NOT suppress.
        score = LONG_STRONG_BULL             # e.g. +65.0
        result = safety.apply_override(score, "sell", LOW_ML)
        assert result["allowed"] is True
        assert result["override_applied"] is False

    def test_well_above_bull_threshold_suppresses_sell(self, safety):
        score = LONG_STRONG_BULL + 10.0      # e.g. +75.0
        result = safety.apply_override(score, "sell", LOW_ML)
        assert result["allowed"] is False
        assert result["override_applied"] is True

    def test_bullish_sell_suppression_with_high_ml_bypassed(self, safety):
        """High ML confidence overrides bullish suppression."""
        score = LONG_STRONG_BULL + 5.0
        result = safety.apply_override(score, "sell", HIGH_ML)
        assert result["allowed"] is True
        assert result["override_applied"] is False

    def test_bullish_does_not_suppress_buy(self, safety):
        """Bullish long trend + short BUY are aligned; no suppression."""
        score = LONG_STRONG_BULL + 5.0
        result = safety.apply_override(score, "buy", LOW_ML)
        assert result["allowed"] is True
        assert result["override_applied"] is False


# ---------------------------------------------------------------------------
# 4. The former 65–70 gap: scores in that band MUST now trigger suppression
# ---------------------------------------------------------------------------

class TestFormerGapNowCovered:
    """
    Regression: before the fix, scores in the ±65–70 band produced a
    confirmed long-trend BUY/SELL but macro suppression never engaged.
    With LONG_STRONG_BULL/BEAR aligned to ±65 the gap is closed.
    """

    @pytest.mark.parametrize("score", [-65.5, -67.0, -69.0, -69.99])
    def test_bear_gap_band_suppresses_conflicting_buy(self, safety, score):
        """Scores in (-70, -65) used to escape suppression; they must not now."""
        result = safety.apply_override(score, "buy", LOW_ML)
        assert result["allowed"] is False, (
            f"score={score} in former gap band should now be suppressed"
        )

    @pytest.mark.parametrize("score", [65.5, 67.0, 69.0, 69.99])
    def test_bull_gap_band_suppresses_conflicting_sell(self, safety, score):
        """Scores in (65, 70) used to escape suppression; they must not now."""
        result = safety.apply_override(score, "sell", LOW_ML)
        assert result["allowed"] is False, (
            f"score={score} in former gap band should now be suppressed"
        )


# ---------------------------------------------------------------------------
# 5. Neutral signals always pass through
# ---------------------------------------------------------------------------

class TestNeutralAlwaysAllowed:
    @pytest.mark.parametrize("score", [
        LONG_STRONG_BEAR - 10,
        0.0,
        LONG_STRONG_BULL + 10,
    ])
    def test_neutral_signal_never_suppressed(self, safety, score):
        result = safety.apply_override(score, "neutral", LOW_ML)
        assert result["allowed"] is True
        assert result["override_applied"] is False


# ---------------------------------------------------------------------------
# 6. Return-value schema
# ---------------------------------------------------------------------------

class TestReturnSchema:
    def test_result_always_has_required_keys(self, safety):
        result = safety.apply_override(0.0, "buy", 0.5)
        assert {"allowed", "override_applied", "reason"} <= result.keys()

    def test_reason_is_non_empty_string(self, safety):
        result = safety.apply_override(LONG_STRONG_BEAR - 1, "buy", LOW_ML)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_bad_inputs_default_to_allowed(self, safety):
        """Invalid inputs must not raise; they should default to allowed=True."""
        result = safety.apply_override("not_a_number", "buy", 0.5)  # type: ignore[arg-type]
        assert result["allowed"] is True
