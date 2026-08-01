"""
Tests for the conviction-tier evaluator and the backtest coverage guardrail.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from signal_engine.conviction import (
    ConvictionEvaluator, TIER_HIGH_CONVICTION, TIER_OPPORTUNITY,
    REGIME_TRANSITION_WINDOW_SECONDS,
)
from scripts.backtest_conviction_tiers import replay, check, load_fixture

FIXTURE = str(Path(__file__).parent / "fixtures" / "conviction_fixture.json")

STRONG_BUY = dict(
    signal_type="buy", gauge_type="long", volatility_regime="calm",
    cycle_quality_score=0.85, ml_confidence=0.90, ml_fallback=False,
    long_score=78.0, short_score=45.0,
)


def make_eval():
    ev = ConvictionEvaluator()
    ev.reset()
    return ev


class TestTierClassification:
    def test_strong_signal_earns_high_conviction(self):
        res = make_eval().evaluate(**STRONG_BUY, now=0.0)
        assert res["tier"] == TIER_HIGH_CONVICTION
        assert any("High conviction" in r for r in res["reasons"])

    def test_neutral_signal_gets_no_tier(self):
        res = make_eval().evaluate(**{**STRONG_BUY, "signal_type": "neutral"}, now=0.0)
        assert res["tier"] is None

    def test_unfavorable_regime_caps_at_opportunity(self):
        res = make_eval().evaluate(
            **{**STRONG_BUY, "volatility_regime": "compressed"}, now=0.0)
        assert res["tier"] == TIER_OPPORTUNITY
        assert any("unfavorable" in r for r in res["reasons"])

    def test_low_cycle_quality_caps_at_opportunity(self):
        res = make_eval().evaluate(
            **{**STRONG_BUY, "cycle_quality_score": 0.5}, now=0.0)
        assert res["tier"] == TIER_OPPORTUNITY

    def test_trend_disagreement_caps_at_opportunity(self):
        res = make_eval().evaluate(**{**STRONG_BUY, "short_score": -20.0}, now=0.0)
        assert res["tier"] == TIER_OPPORTUNITY

    def test_ml_fallback_never_earns_high_conviction(self):
        res = make_eval().evaluate(**{**STRONG_BUY, "ml_fallback": True}, now=0.0)
        assert res["tier"] == TIER_OPPORTUNITY
        assert any("unavailable" in r for r in res["reasons"])

    def test_sell_uses_directional_ml_confidence(self):
        # ml_confidence is a BUY probability: 0.1 is a CONFIDENT sell.
        res = make_eval().evaluate(
            signal_type="sell", gauge_type="long", volatility_regime="calm",
            cycle_quality_score=0.8, ml_confidence=0.10, ml_fallback=False,
            long_score=-75.0, short_score=-40.0, now=0.0)
        assert res["tier"] == TIER_HIGH_CONVICTION

    def test_never_suppresses_actionable_signal(self):
        # Worst possible inputs still get a tier — labeling only.
        res = make_eval().evaluate(
            signal_type="buy", gauge_type="short",
            volatility_regime="macro_shock", cycle_quality_score=0.0,
            ml_confidence=0.0, ml_fallback=True,
            long_score=-90.0, short_score=-90.0, now=0.0)
        assert res["tier"] == TIER_OPPORTUNITY


class TestRegimeTransitionDowngrade:
    def test_recent_regime_change_caps_tier(self):
        ev = make_eval()
        ev.evaluate(**STRONG_BUY, now=0.0)                       # calm observed
        res = ev.evaluate(
            **{**STRONG_BUY, "volatility_regime": "trending"}, now=100.0)
        assert res["tier"] == TIER_OPPORTUNITY
        assert any("regime shifted" in r.lower() for r in res["reasons"])

    def test_transition_window_expires(self):
        ev = make_eval()
        ev.evaluate(**STRONG_BUY, now=0.0)
        ev.evaluate(**{**STRONG_BUY, "volatility_regime": "trending"}, now=100.0)
        later = 100.0 + REGIME_TRANSITION_WINDOW_SECONDS + 1
        res = ev.evaluate(
            **{**STRONG_BUY, "volatility_regime": "trending"}, now=later)
        assert res["tier"] == TIER_HIGH_CONVICTION

    def test_first_observation_is_not_a_transition(self):
        res = make_eval().evaluate(**STRONG_BUY, now=0.0)
        assert res["tier"] == TIER_HIGH_CONVICTION


class TestHysteresis:
    def test_small_dip_keeps_high_conviction(self):
        ev = make_eval()
        assert ev.evaluate(**STRONG_BUY, now=0.0)["tier"] == TIER_HIGH_CONVICTION
        # Quality dips below entry (0.65) but stays above exit (0.55)
        res = ev.evaluate(
            **{**STRONG_BUY, "cycle_quality_score": 0.60}, now=60.0)
        assert res["tier"] == TIER_HIGH_CONVICTION

    def test_decisive_drop_demotes(self):
        ev = make_eval()
        assert ev.evaluate(**STRONG_BUY, now=0.0)["tier"] == TIER_HIGH_CONVICTION
        res = ev.evaluate(
            **{**STRONG_BUY, "cycle_quality_score": 0.40}, now=60.0)
        assert res["tier"] == TIER_OPPORTUNITY

    def test_borderline_signal_does_not_flicker(self):
        ev = make_eval()
        assert ev.evaluate(**STRONG_BUY, now=0.0)["tier"] == TIER_HIGH_CONVICTION
        # Oscillate quality around the entry threshold: tier must stay put.
        for i, q in enumerate((0.63, 0.67, 0.62, 0.66), start=1):
            res = ev.evaluate(
                **{**STRONG_BUY, "cycle_quality_score": q}, now=60.0 * i)
            assert res["tier"] == TIER_HIGH_CONVICTION

    def test_promotion_requires_full_entry_criteria(self):
        ev = make_eval()
        # Starts as opportunity (quality 0.60 < entry 0.65)
        assert ev.evaluate(
            **{**STRONG_BUY, "cycle_quality_score": 0.60}, now=0.0
        )["tier"] == TIER_OPPORTUNITY
        # Still below entry → stays opportunity (no upward flicker either)
        assert ev.evaluate(
            **{**STRONG_BUY, "cycle_quality_score": 0.63}, now=60.0
        )["tier"] == TIER_OPPORTUNITY
        # Clears entry → promoted
        assert ev.evaluate(
            **{**STRONG_BUY, "cycle_quality_score": 0.70}, now=120.0
        )["tier"] == TIER_HIGH_CONVICTION


class TestBacktestGuardrail:
    def test_fixture_replay_passes_guardrails(self):
        signals = load_fixture(FIXTURE)
        report = replay(signals)
        failures = check(report)
        assert failures == [], failures

    def test_full_coverage_no_signal_lost(self):
        signals = load_fixture(FIXTURE)
        report = replay(signals)
        actionable = report["input_signals"] - report["untiered_neutral"]
        assert report["overall"]["signals"] == actionable

    def test_high_conviction_outperforms_per_cycle(self):
        report = replay(load_fixture(FIXTURE))
        assert report["high_conviction"]["avg_return"] > report["overall"]["avg_return"]
        assert report["high_conviction"]["win_rate"] > report["overall"]["win_rate"]

    def test_guardrail_detects_coverage_loss(self):
        # Synthetic broken report: half the actionable signals lost a tier.
        report = {
            "input_signals": 10, "untiered_neutral": 0,
            "overall": {"signals": 5, "cycles": 5, "win_rate": 0.6, "avg_return": 0.5},
            "high_conviction": {"signals": 2, "cycles": 2, "win_rate": 1.0, "avg_return": 1.5},
            "opportunity_only": {"signals": 3, "cycles": 3, "win_rate": 0.3, "avg_return": -0.2},
        }
        failures = check(report)
        assert any("coverage guardrail" in f for f in failures)

    def test_guardrail_detects_underperforming_high_conviction(self):
        report = {
            "input_signals": 10, "untiered_neutral": 0,
            "overall": {"signals": 10, "cycles": 10, "win_rate": 0.6, "avg_return": 0.8},
            "high_conviction": {"signals": 4, "cycles": 4, "win_rate": 0.5, "avg_return": 0.2},
            "opportunity_only": {"signals": 6, "cycles": 6, "win_rate": 0.7, "avg_return": 1.2},
        }
        failures = check(report)
        assert any("profitability" in f for f in failures)
