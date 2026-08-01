"""
Unit tests: HoldTimePredictionEngine – BUY threshold alignment
==============================================================
Confirms that the hold-time engine uses LONG_BUY_THRESHOLD (65.0) as its
long-trigger cutoff, not the old hardcoded 70.0.

Key scenario under test:
  A long_score in the 65–70 band is above LONG_BUY_THRESHOLD (so the gauge
  emits a BUY signal) and must therefore trigger the 15-day base hold-time,
  not fall back to the 4-hour default.
"""

import pytest

from ml.hold_time import HoldTimePredictionEngine
from config import settings

# Minimal indicators dict – no ADX or Bollinger adjustments
_NO_INDICATORS: dict = {"latest": {}}


class TestLongTriggerThreshold:
    """long_triggered must fire at LONG_BUY_THRESHOLD, not 70.0."""

    def setup_method(self):
        self.engine = HoldTimePredictionEngine()

    # ------------------------------------------------------------------
    # Band that was broken: 65 < score ≤ 70
    # ------------------------------------------------------------------

    def test_score_at_threshold_triggers_long_hold(self):
        """
        A score exactly at LONG_BUY_THRESHOLD + ε (just above 65) must yield
        a 15-day base hold-time (after multipliers the result is ≥ 1 day).
        """
        score = settings.LONG_BUY_THRESHOLD + 0.01   # e.g. 65.01
        result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=score,
            short_score=0.0,
            vix_regime="NORMAL",
        )
        # 15 days = 21 600 min; NORMAL VIX keeps multiplier at 1.0
        assert result["minutes"] == pytest.approx(
            HoldTimePredictionEngine._BASE_LONG_MINUTES, rel=0.01
        ), (
            f"score={score} should trigger long base (21600 min) but got "
            f"{result['minutes']} min. reasoning={result['reasoning']}"
        )

    def test_score_in_65_70_band_triggers_long_hold(self):
        """
        A score of 67.5 is above LONG_BUY_THRESHOLD (65) → 15-day hold.
        Under the old hardcoded 70.0 cutoff this would have fallen through
        to the 4-hour default.
        """
        result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=67.5,
            short_score=0.0,
            vix_regime="NORMAL",
        )
        assert result["minutes"] == pytest.approx(
            HoldTimePredictionEngine._BASE_LONG_MINUTES, rel=0.01
        ), (
            f"score=67.5 should trigger long base but got {result['minutes']} min. "
            f"reasoning={result['reasoning']}"
        )

    def test_score_just_below_threshold_does_not_trigger_long_hold(self):
        """
        A score just below LONG_BUY_THRESHOLD (65) must NOT trigger the long
        hold-time.  Both long and short scores are below their thresholds here,
        so the result should be the 4-hour default.
        """
        score = settings.LONG_BUY_THRESHOLD - 0.01   # e.g. 64.99
        result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=score,
            short_score=0.0,
            vix_regime="NORMAL",
        )
        assert result["minutes"] == pytest.approx(
            HoldTimePredictionEngine._BASE_DEFAULT_MINUTES, rel=0.01
        ), (
            f"score={score} is below LONG_BUY_THRESHOLD; expected default "
            f"({HoldTimePredictionEngine._BASE_DEFAULT_MINUTES} min) but got "
            f"{result['minutes']} min."
        )

    def test_score_above_70_still_triggers_long_hold(self):
        """Scores above 70 (the old cutoff) must still work correctly."""
        result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=80.0,
            short_score=0.0,
            vix_regime="NORMAL",
        )
        assert result["minutes"] == pytest.approx(
            HoldTimePredictionEngine._BASE_LONG_MINUTES, rel=0.01
        )

    # ------------------------------------------------------------------
    # Threshold comes from config, not hardcoded
    # ------------------------------------------------------------------

    def test_long_triggered_uses_config_threshold(self):
        """
        The threshold used by the engine must equal settings.LONG_BUY_THRESHOLD.
        Verify by probing at LONG_BUY_THRESHOLD + 0.01 (should trigger) and
        at LONG_BUY_THRESHOLD - 0.01 (should not trigger).
        """
        just_above = settings.LONG_BUY_THRESHOLD + 0.01
        just_below = settings.LONG_BUY_THRESHOLD - 0.01

        above_result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=just_above,
            short_score=0.0,
            vix_regime="NORMAL",
        )
        below_result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=just_below,
            short_score=0.0,
            vix_regime="NORMAL",
        )

        assert above_result["minutes"] > below_result["minutes"], (
            "A score just above LONG_BUY_THRESHOLD should produce a longer "
            "hold-time than a score just below it."
        )
        assert above_result["minutes"] == pytest.approx(
            HoldTimePredictionEngine._BASE_LONG_MINUTES, rel=0.01
        )
        assert below_result["minutes"] == pytest.approx(
            HoldTimePredictionEngine._BASE_DEFAULT_MINUTES, rel=0.01
        )

    # ------------------------------------------------------------------
    # Reasoning strings
    # ------------------------------------------------------------------

    def test_reasoning_mentions_long_trend_for_65_70_score(self):
        """Reasoning list must mention the long-trend signal for a 65–70 score."""
        result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=67.0,
            short_score=0.0,
            vix_regime="NORMAL",
        )
        combined = " ".join(result["reasoning"]).lower()
        assert "long" in combined and "15 day" in combined, (
            f"Expected reasoning to mention 'long' and '15 day', got: {result['reasoning']}"
        )


class TestShortTriggerThresholdUnchanged:
    """SHORT_BUY_THRESHOLD (50.0) must still work as before."""

    def setup_method(self):
        self.engine = HoldTimePredictionEngine()

    def test_short_score_above_50_triggers_short_hold(self):
        result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=0.0,
            short_score=55.0,
            vix_regime="NORMAL",
        )
        assert result["minutes"] == pytest.approx(
            HoldTimePredictionEngine._BASE_SHORT_MINUTES, rel=0.01
        )

    def test_both_triggered_long_dominates(self):
        """When both gauges fire, the long (15-day) base must win."""
        result = self.engine.estimate_hold_time(
            indicators=_NO_INDICATORS,
            long_score=67.0,   # in the 65–70 band
            short_score=55.0,
            vix_regime="NORMAL",
        )
        assert result["minutes"] == pytest.approx(
            HoldTimePredictionEngine._BASE_LONG_MINUTES, rel=0.01
        )
