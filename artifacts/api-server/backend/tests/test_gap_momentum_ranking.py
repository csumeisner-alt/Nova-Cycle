"""
Tests: gap follow-through (gap_momentum) influence on the short gauge.

Behaviour is additive:
  - gap_momentum=None → identical to legacy behaviour
  - follow-through (momentum >= +GAP_MOMENTUM_THRESHOLD) boosts the score
    toward the gap direction
  - fade (momentum <= -GAP_MOMENTUM_THRESHOLD) pushes the score away from
    the gap direction
  - small momentum inside the dead zone has no effect
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from signal_engine.short_gauge import ShortTrendGauge
from routers.predictions import _compute_gap_momentum_from_df

GAUGE = ShortTrendGauge()
NEUTRAL_IND = {"latest": {"rsi": 50.0, "stoch_rsi_k": 50.0, "stoch_k": 50.0, "bb_pct_b": 0.5}}
BOOST = settings.GAP_MOMENTUM_SCORE_BOOST
THRESH = settings.GAP_MOMENTUM_THRESHOLD


def _score(gap_type, gap_momentum):
    total, breakdown = GAUGE.compute_indicator_score(
        NEUTRAL_IND, is_extended=False, liquidity_score=1.0,
        gap_type=gap_type, gap_momentum=gap_momentum,
    )
    return total, breakdown


class TestMomentumInfluence:
    def test_none_momentum_is_noop(self):
        base, _ = _score("gap_up", None)
        legacy, _ = GAUGE.compute_indicator_score(
            NEUTRAL_IND, is_extended=False, liquidity_score=1.0, gap_type="gap_up"
        )
        assert base == legacy

    def test_gap_up_follow_through_boosts_bullish(self):
        base, _ = _score("gap_up", None)
        total, bd = _score("gap_up", THRESH + 0.5)
        assert total == base + BOOST
        assert "follow-through" in bd["gap_momentum_influence"]

    def test_gap_up_fade_downgrades(self):
        base, _ = _score("gap_up", None)
        total, bd = _score("gap_up", -(THRESH + 0.5))
        assert total == base - BOOST
        assert "fading" in bd["gap_momentum_influence"]

    def test_gap_down_follow_through_boosts_bearish(self):
        base, _ = _score("gap_down", None)
        total, _ = _score("gap_down", THRESH + 0.5)
        assert total == base - BOOST  # more bearish

    def test_gap_down_fade_downgrades(self):
        base, _ = _score("gap_down", None)
        total, _ = _score("gap_down", -(THRESH + 0.5))
        assert total == base + BOOST  # less bearish

    def test_dead_zone_no_effect(self):
        base, _ = _score("gap_up", None)
        total, bd = _score("gap_up", THRESH / 2.0)
        assert total == base
        assert "gap_momentum_influence" not in bd

    def test_no_gap_no_effect(self):
        base, _ = _score("none", None)
        total, bd = _score("none", 1.0)
        assert total == base
        assert "gap_momentum_influence" not in bd

    def test_compute_score_carries_gap_momentum(self):
        res = GAUGE.compute_score(
            NEUTRAL_IND, ml_prediction=0.5, is_extended=False,
            liquidity_score=1.0, gap_type="gap_up", gap_momentum=0.7,
        )
        assert res["gap_momentum"] == 0.7

    def test_compute_score_confidence_reflects_momentum(self):
        """Follow-through raises confidence for aligned signals; fade lowers it.

        Confidence feeds the existing per-device sensitivity thresholds in
        the notification path, so this is what makes the behaviour
        configurable via existing settings.
        """
        bullish = {"latest": {"rsi": 25.0, "stoch_rsi_k": 10.0, "stoch_k": 10.0, "bb_pct_b": -0.1}}
        follow = GAUGE.compute_score(bullish, 0.9, False, 1.0, "gap_up", gap_momentum=1.0)
        fade = GAUGE.compute_score(bullish, 0.9, False, 1.0, "gap_up", gap_momentum=-1.0)
        assert follow["score"] > fade["score"]
        assert follow["confidence"] > fade["confidence"]


class TestComputeGapMomentumFromDf:
    def _frame(self, gap_pct=2.0, n_regular=6, drift=0.1):
        rows = []
        base = pd.Timestamp("2026-07-24 08:00:00")
        rows.append({"timestamp": base, "open": 100.0, "close": 100.0,
                     "session_type": "pre_market", "gap_percent": gap_pct})
        px = 100.0
        for i in range(n_regular):
            ts = pd.Timestamp("2026-07-24 09:30:00") + pd.Timedelta(minutes=5 * i)
            rows.append({"timestamp": ts, "open": px, "close": px + drift,
                         "session_type": "regular", "gap_percent": 0.0})
            px += drift
        return pd.DataFrame(rows)

    def test_follow_through_positive(self):
        m = _compute_gap_momentum_from_df(self._frame(gap_pct=2.0, drift=0.1))
        assert m is not None and m > 0

    def test_fade_negative(self):
        m = _compute_gap_momentum_from_df(self._frame(gap_pct=2.0, drift=-0.1))
        assert m is not None and m < 0

    def test_not_enough_candles_returns_none(self):
        assert _compute_gap_momentum_from_df(self._frame(n_regular=3)) is None

    def test_no_gap_returns_none(self):
        df = self._frame()
        df = df[df["session_type"] == "regular"]
        assert _compute_gap_momentum_from_df(df) is None

    def test_empty_df_returns_none(self):
        assert _compute_gap_momentum_from_df(pd.DataFrame()) is None
