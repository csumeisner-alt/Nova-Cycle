"""
Tests for the VOO-only decision-layer filter.

These tests verify the five decision-layer upgrades:
  1. Volatility regime filtering
  2. Gap-type filtering
  3. Liquidity-class filtering
  4. Confidence divergence suppression
  5. Cycle-quality scoring

They do not require live data, network access, or a trained model.
"""

import pytest

from signal_engine.decision_filter import DecisionFilter
from config import settings


@pytest.fixture
def df():
    return DecisionFilter()


def _base_history(**overrides):
    """Return a minimal confidence history list; override values as needed."""
    return [{
        "long_buy_confidence": overrides.get("long_buy_confidence", 0.5),
        "long_sell_confidence": 0.5,
        "short_buy_confidence": overrides.get("short_buy_confidence", 0.5),
        "short_sell_confidence": 0.5,
    }]


def _evaluate(df, signal_type, **overrides):
    """Convenience wrapper for DecisionFilter.evaluate with sensible defaults."""
    kwargs = {
        "signal_type": signal_type,
        "score": 75.0,
        "ml_confidence": 0.8,
        "indicators": {"latest": {"vix_regime": "NORMAL", "atr_compression_score": 0.3, "trend_strength_index": 0.5}},
        "latest_candle": {"gap_type": "none", "gap_percent": 0.0},
        "liquidity_score": 1.0,
        "gap_momentum": None,
        "confidence_history": _base_history(),
    }
    kwargs.update(overrides)
    return df.evaluate(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Volatility regime filtering
# ─────────────────────────────────────────────────────────────────────────────

def test_buy_blocked_in_macro_shock(df):
    indicators = {"latest": {"vix_regime": "EXTREME", "atr_compression_score": 0.3, "trend_strength_index": 0.5}}
    result = _evaluate(df, "buy", indicators=indicators)
    assert result["final_signal"] == "neutral"
    assert result["volatility_regime"] == "macro_shock"
    assert "BUY blocked" in result["reason"]


def test_buy_is_kept_as_opportunity_in_compressed_regime(df):
    indicators = {"latest": {"vix_regime": "LOW", "atr_compression_score": 0.1, "trend_strength_index": 0.5}}
    result = _evaluate(df, "buy", indicators=indicators)
    assert result["final_signal"] == "buy"
    assert result["volatility_regime"] == "compressed"
    assert result["conviction_tier_cap"] == "opportunity"


def test_buy_allowed_in_calm_regime(df):
    result = _evaluate(df, "buy")
    assert result["final_signal"] == "buy"
    assert result["volatility_regime"] == "calm"


def test_sell_allowed_in_macro_shock_with_priority_boost(df):
    indicators = {"latest": {"vix_regime": "EXTREME", "atr_compression_score": 0.3, "trend_strength_index": 0.5}}
    result = _evaluate(df, "sell", indicators=indicators)
    assert result["final_signal"] == "sell"
    assert result["volatility_regime"] == "macro_shock"
    assert result["priority_boost"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gap-type filtering
# ─────────────────────────────────────────────────────────────────────────────

def test_buy_is_kept_as_opportunity_after_negative_macro_gap(df):
    latest_candle = {"gap_type": "gap_down", "gap_percent": -1.5}
    result = _evaluate(df, "buy", latest_candle=latest_candle)
    assert result["final_signal"] == "buy"
    assert "negative macro gap" in result["reason"]
    assert result["conviction_tier_cap"] == "opportunity"


def test_buy_prioritized_after_positive_continuation_gap(df):
    latest_candle = {"gap_type": "gap_up", "gap_percent": 1.5}
    result = _evaluate(df, "buy", latest_candle=latest_candle, gap_momentum=0.15)
    assert result["final_signal"] == "buy"
    assert result["priority_boost"] > 0.0


def test_sell_blocked_during_strong_positive_gap_without_flip(df):
    latest_candle = {"gap_type": "gap_up", "gap_percent": 1.5}
    history = [
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.5},
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.55},
    ]
    result = _evaluate(df, "sell", latest_candle=latest_candle, confidence_history=history)
    assert result["final_signal"] == "neutral"
    assert "strong positive continuation gap" in result["reason"]


def test_sell_allowed_during_strong_positive_gap_when_momentum_flips(df):
    latest_candle = {"gap_type": "gap_up", "gap_percent": 1.5}
    history = [
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.55},
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.45},
    ]
    result = _evaluate(df, "sell", latest_candle=latest_candle, confidence_history=history)
    assert result["final_signal"] == "sell"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Liquidity-class filtering
# ─────────────────────────────────────────────────────────────────────────────

def test_buy_is_kept_as_opportunity_in_moderate_low_liquidity(df):
    result = _evaluate(df, "buy", liquidity_score=0.3)
    assert result["final_signal"] == "buy"
    assert result["liquidity_class"] == "low"
    assert "low liquidity" in result["reason"]
    assert result["conviction_tier_cap"] == "opportunity"


def test_buy_blocked_in_extremely_low_liquidity(df):
    result = _evaluate(df, "buy", liquidity_score=0.1)
    assert result["final_signal"] == "neutral"
    assert "extremely low liquidity" in result["reason"]


def test_buy_allowed_in_normal_liquidity(df):
    result = _evaluate(df, "buy", liquidity_score=0.6)
    assert result["final_signal"] == "buy"
    assert result["liquidity_class"] == "normal"


def test_buy_allowed_in_high_liquidity(df):
    result = _evaluate(df, "buy", liquidity_score=1.2)
    assert result["final_signal"] == "buy"
    assert result["liquidity_class"] == "high"


def test_sell_priority_increased_in_low_liquidity(df):
    result = _evaluate(df, "sell", liquidity_score=0.3)
    assert result["final_signal"] == "sell"
    assert result["liquidity_class"] == "low"
    assert result["priority_boost"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Confidence divergence suppression
# ─────────────────────────────────────────────────────────────────────────────

def test_buy_is_kept_as_opportunity_when_long_rises_and_short_falls(df):
    history = [
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.7},
        {"long_buy_confidence": 0.6, "short_buy_confidence": 0.6},
    ]
    result = _evaluate(df, "buy", confidence_history=history)
    assert result["final_signal"] == "buy"
    assert result["filter_flags"]["divergence"] is True
    assert result["conviction_tier_cap"] == "opportunity"


def test_buy_allowed_when_confidence_rises_together(df):
    history = [
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.5},
        {"long_buy_confidence": 0.6, "short_buy_confidence": 0.6},
    ]
    result = _evaluate(df, "buy", confidence_history=history)
    assert result["final_signal"] == "buy"
    assert result["filter_flags"]["divergence"] is False


def test_buy_is_kept_as_opportunity_when_confidence_momentum_negative(df):
    history = [
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.6},
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.5},
    ]
    result = _evaluate(df, "buy", confidence_history=history)
    assert result["final_signal"] == "buy"
    assert result["confidence_momentum"] < 0
    assert result["conviction_tier_cap"] == "opportunity"


def test_sell_allowed_when_confidence_momentum_flips(df):
    history = [
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.6},
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.5},
    ]
    result = _evaluate(df, "sell", confidence_history=history)
    assert result["final_signal"] == "sell"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cycle-quality scoring
# ─────────────────────────────────────────────────────────────────────────────

def test_cycle_quality_score_combines_factors(df):
    score = df.compute_cycle_quality_score(
        volatility_regime="calm",
        gap_type="gap_up",
        gap_percent=1.5,
        gap_momentum=0.15,
        liquidity_class="high",
        confidence_momentum=0.1,
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_buy_is_kept_as_opportunity_when_cycle_quality_below_threshold(df):
    # Low liquidity + compressed regime pushes score below 0.6
    indicators = {"latest": {"vix_regime": "LOW", "atr_compression_score": 0.1, "trend_strength_index": 0.5}}
    result = _evaluate(df, "buy", indicators=indicators, liquidity_score=0.3)
    assert result["final_signal"] == "buy"
    assert result["cycle_quality_score"] < settings.DECISION_BUY_MIN_CYCLE_QUALITY
    assert result["conviction_tier_cap"] == "opportunity"


def test_sell_allowed_regardless_of_cycle_quality(df):
    indicators = {"latest": {"vix_regime": "LOW", "atr_compression_score": 0.1, "trend_strength_index": 0.5}}
    result = _evaluate(df, "sell", indicators=indicators, liquidity_score=0.3)
    assert result["final_signal"] == "sell"
    assert result["priority_boost"] > 0.0
    assert result["conviction_tier_cap"] == "opportunity"


def test_sell_priority_increased_when_cycle_quality_low(df):
    indicators = {"latest": {"vix_regime": "LOW", "atr_compression_score": 0.1, "trend_strength_index": 0.5}}
    result = _evaluate(df, "sell", indicators=indicators, liquidity_score=0.3)
    assert result["cycle_quality_score"] < settings.DECISION_BUY_MIN_CYCLE_QUALITY
    assert result["priority_boost"] > 0.0
    assert result["conviction_tier_cap"] == "opportunity"


def test_degraded_data_blocks_actionable_signal(df):
    result = _evaluate(df, "buy", data_quality_degraded=True)
    assert result["final_signal"] == "neutral"
    assert "data quality" in result["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper method unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_infer_volatility_regime_defaults_to_calm(df):
    assert df.infer_volatility_regime({"latest": {}}) == "calm"


def test_infer_volatility_regime_from_vix_extreme(df):
    indicators = {"latest": {"vix_regime": "EXTREME", "atr_compression_score": 0.3, "trend_strength_index": 0.5}}
    assert df.infer_volatility_regime(indicators) == "macro_shock"


def test_classify_liquidity(df):
    assert df.classify_liquidity(1.2) == "high"
    assert df.classify_liquidity(0.7) == "normal"
    assert df.classify_liquidity(0.3) == "low"


def test_compute_confidence_metrics(df):
    history = [
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.6},
        {"long_buy_confidence": 0.6, "short_buy_confidence": 0.5},
    ]
    long_conf, short_conf, momentum = df.compute_confidence_metrics(history)
    assert long_conf == pytest.approx(0.6)
    assert short_conf == pytest.approx(0.5)
    assert momentum == pytest.approx(-0.1)


def test_compute_confidence_metrics_ignores_zeros(df):
    history = [
        {"long_buy_confidence": 0.0, "short_buy_confidence": 0.6},
        {"long_buy_confidence": 0.5, "short_buy_confidence": 0.0},
    ]
    long_conf, short_conf, momentum = df.compute_confidence_metrics(history)
    assert long_conf == 0.5
    assert short_conf == 0.6
    assert momentum == 0.0


def test_neutral_signal_passthrough(df):
    result = df.evaluate(
        signal_type="neutral",
        score=0.0,
        ml_confidence=0.5,
        indicators={},
        latest_candle={},
        liquidity_score=1.0,
        gap_momentum=None,
        confidence_history=[],
    )
    assert result["allowed"] is True
    assert result["final_signal"] == "neutral"
