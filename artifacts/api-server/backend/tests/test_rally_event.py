"""Tests for the shared rally-event definition (rally_event.py).

The training label, walk-forward evaluation, and missed-rally reporting must
all describe the same event: a >0.3% rise from the observation close at ANY
point within the next 12 five-minute bars.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from rally_event import (
    RALLY_HORIZON_BARS,
    RALLY_RISE_FRACTION,
    RALLY_RISE_PERCENT,
    rally_event_labels,
)
from performance_engine import (
    MISSED_RALLY_BARS,
    MISSED_RALLY_RISE_PERCENT,
    find_all_missed_rallies_in_candles,
)


def _series(closes):
    idx = pd.date_range("2026-08-03 13:30", periods=len(closes), freq="5min")
    return pd.Series(closes, index=idx, dtype=float)


class TestSharedConstants:
    def test_missed_rally_detector_uses_shared_definition(self):
        assert MISSED_RALLY_BARS == RALLY_HORIZON_BARS
        assert MISSED_RALLY_RISE_PERCENT == RALLY_RISE_PERCENT


class TestRallyEventLabels:
    def test_spike_and_retrace_within_window_is_positive(self):
        # Price pops +0.5% three bars ahead, then fully retraces before bar
        # 12.  The old exact-12-bar label called this negative; the shared
        # window-max definition calls it a rally.
        closes = [100.0, 100.0, 100.0, 100.5] + [100.0] * 20
        labels = rally_event_labels(_series(closes))
        assert labels.iloc[0] == 1.0

    def test_flat_series_is_negative(self):
        labels = rally_event_labels(_series([100.0] * 30))
        valid = labels.dropna()
        assert len(valid) > 0
        assert (valid == 0.0).all()

    def test_rise_just_below_threshold_is_negative(self):
        closes = [100.0] + [100.0 * (1 + RALLY_RISE_FRACTION)] * 20
        labels = rally_event_labels(_series(closes))
        # exactly at threshold is NOT a rally (strict >)
        assert labels.iloc[0] == 0.0

    def test_tail_rows_without_full_window_are_nan(self):
        labels = rally_event_labels(_series([100.0] * 30))
        assert labels.iloc[-RALLY_HORIZON_BARS:].isna().all()
        assert labels.iloc[: 30 - RALLY_HORIZON_BARS].notna().all()

    def test_empty_series(self):
        assert rally_event_labels(pd.Series(dtype=float)).empty

    def test_labels_agree_with_missed_rally_detector(self):
        # Build a window with two distinct rally episodes and verify that
        # every detector hit corresponds to a positive label at that bar.
        rng = np.random.default_rng(7)
        closes = list(100 + rng.normal(0, 0.01, 40).cumsum())
        closes[10] = closes[9] * 1.005   # rally episode 1
        closes[30] = closes[29] * 1.006  # rally episode 2
        s = _series(closes)
        labels = rally_event_labels(s)
        candles = list(zip(s.index.to_pydatetime(), s.values))
        hits = find_all_missed_rallies_in_candles(candles)
        assert len(hits) >= 2
        positions = {ts: i for i, ts in enumerate(s.index.to_pydatetime())}
        for ts in hits:
            i = positions[ts]
            assert labels.iloc[i] == 1.0, f"detector hit at {ts} not labeled positive"


class TestFeatureLabelAlignment:
    def test_skipped_mid_series_row_does_not_shift_labels(self):
        """A bad (close <= 0) row in the middle of the training frame must be
        dropped from BOTH features and labels — not silently shift y."""
        from ml.short_trend import ShortTrendModel

        n = 60
        closes = [100.0] * n
        closes[25] = -1.0  # invalid row build_features skips
        idx = pd.date_range("2026-08-03 13:30", periods=n, freq="5min")
        df = pd.DataFrame(
            {"open": closes, "close": closes, "volume": [1000.0] * n}, index=idx
        )
        df["label"] = rally_event_labels(df["close"])
        df2 = df.dropna(subset=["label"]).copy()
        df2["label"] = df2["label"].astype(int)

        model = ShortTrendModel()
        X, w, valid_pos = model.build_features(df2, indicators={})
        skipped = [p for p in range(len(df2)) if float(df2["close"].iloc[p]) <= 0]
        kept = [p for p in range(len(df2)) if p not in skipped]
        assert len(X) == len(w) == len(valid_pos) == len(kept)
        assert set(valid_pos.tolist()) == set(kept)
        y = df2["label"].values[valid_pos]
        assert len(y) == len(X)
        # labels selected by valid_pos must match the surviving frame rows
        expected = df2["label"].values[kept]
        assert (y == expected).all()


class TestShortEventGate:
    def test_gate_fails_when_pr_auc_at_or_below_base_rate(self):
        from ml.trainer import _short_event_gate_failed

        bad = {"walk_forward": {"evaluated": True, "pr_auc": 0.20, "positive_rate": 0.25}}
        assert _short_event_gate_failed(bad) is not None

    def test_gate_passes_with_ranking_power(self):
        from ml.trainer import _short_event_gate_failed

        good = {"walk_forward": {"evaluated": True, "pr_auc": 0.45, "positive_rate": 0.25}}
        assert _short_event_gate_failed(good) is None

    def test_gate_skipped_when_not_evaluated(self):
        from ml.trainer import _short_event_gate_failed

        assert _short_event_gate_failed({"walk_forward": {"evaluated": False}}) is None
        assert _short_event_gate_failed({}) is None
