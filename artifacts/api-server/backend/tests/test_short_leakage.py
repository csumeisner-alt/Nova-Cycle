"""
Short-trend leakage fixes: causal features, purged walk-forward training,
honest reported accuracy, and metric-aware regression checks.

Guards against reintroducing:
  - features that read data after bar t (label covers (t, t+12])
  - a feature scaler fitted on validation/test rows
  - an unpurged split at the train/validation boundary
  - the leakage-inflated 98%+ "accuracy" being reported/persisted as honest
"""

import numpy as np
import pandas as pd
import pytest

from ml import calibration as cal
from ml import short_trend as st
from ml.short_trend import ScaledMLP, ShortTrendModel, LABEL_HORIZON_BARS


def _make_5min_df(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-01 09:30", periods=n, freq="5min")
    steps = rng.normal(loc=0.0, scale=0.4, size=n)
    close = 560 + np.cumsum(steps)
    df = pd.DataFrame({
        "open": close - rng.uniform(0, 0.3, n),
        "high": close + rng.uniform(0, 0.4, n),
        "low": close - rng.uniform(0, 0.4, n),
        "close": close,
        "volume": rng.integers(10_000, 80_000, n).astype(float),
        "is_extended_hours": False,
    }, index=idx)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Leakage audit: features must be causal (no future data)
# ─────────────────────────────────────────────────────────────────────────────

def test_features_are_causal_truncation_invariance():
    """Feature rows up to bar t must be identical whether or not the frame
    contains data after t. Any feature that peeks past t (into the label
    window) breaks this invariance."""
    df = _make_5min_df()
    model = ShortTrendModel()

    X_full, _ = model.build_features(df, indicators={})
    cut = len(df) - 100
    X_trunc, _ = model.build_features(df.iloc[:cut], indicators={})

    assert len(X_trunc) == cut
    np.testing.assert_allclose(
        X_full[:cut], X_trunc, rtol=0, atol=1e-9,
        err_msg="a short-trend feature reads data after bar t (look-ahead leakage)",
    )


def test_label_horizon_constant_matches_target():
    # The 1h label = 12 bars at 5-min resolution; the embargo must cover it.
    assert LABEL_HORIZON_BARS == 12


# ─────────────────────────────────────────────────────────────────────────────
# Scaler leakage: fold-local fitting
# ─────────────────────────────────────────────────────────────────────────────

def test_scaled_mlp_fits_scaler_only_on_training_rows():
    rng = np.random.default_rng(1)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(300, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    p = ScaledMLP()
    p.fit(X_train, y_train)
    # Scaler statistics must reflect the training rows only
    np.testing.assert_allclose(p.scaler.mean_, X_train.mean(axis=0), atol=1e-9)
    # And transform+predict must work on unseen (shifted) data
    X_new = rng.normal(loc=5.0, scale=1.0, size=(10, 4))
    probs = p.predict_proba(X_new)
    assert probs.shape == (10, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Purged walk-forward split
# ─────────────────────────────────────────────────────────────────────────────

def test_walk_forward_uses_label_horizon_embargo():
    df = _make_5min_df()
    seen_train_sizes = []

    class Recorder:
        def fit(self, X, y, sample_weight=None, verbose=False):
            seen_train_sizes.append(len(X))
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    model = ShortTrendModel()
    X, w = model.build_features(df, indicators={})
    y = np.zeros(len(X), dtype=int)
    y[::3] = 1
    metrics, _, _ = cal.walk_forward_evaluate(
        X, y, w, model_factory=Recorder, embargo=LABEL_HORIZON_BARS
    )
    assert metrics["evaluated"] is True
    assert metrics["embargo_rows"] == LABEL_HORIZON_BARS
    n = len(X)
    test_start = max(max(100, LABEL_HORIZON_BARS * 3) + LABEL_HORIZON_BARS, int(n * 0.5))
    edges = np.linspace(test_start, n, metrics["n_splits"] + 1, dtype=int)
    for train_rows, t0 in zip(seen_train_sizes, edges[:-1]):
        assert train_rows <= int(t0) - LABEL_HORIZON_BARS


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end training: honest metrics, non-degenerate live probabilities
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(st, "MODEL_PATH", tmp_path / "short_trend_model.pkl")
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    return tmp_path


def test_train_reports_honest_oos_accuracy(isolated_model_dir):
    df = _make_5min_df(n=900, seed=3)
    m = ShortTrendModel()
    result = m.train(df, indicators={})

    assert result.get("accuracy_metric") == "purged_walk_forward_oos"
    wf = result["walk_forward"]
    assert wf["evaluated"] is True
    # Honest OOS accuracy on near-random 5-min returns must be realistic,
    # never the leakage-inflated 95%+ range.
    assert 0.0 <= result["accuracy"] <= 0.90
    assert result["accuracy"] == pytest.approx(wf["oos_accuracy"])
    # train accuracy is still reported separately for transparency
    assert "train_accuracy" in result

    # Walk-forward report persisted for healthz
    report = cal.get_walkforward_report("short_trend")
    assert report is not None and report["evaluated"] is True

    # Live inference: non-degenerate probabilities across recent bars
    X, _ = m.build_features(df, indicators={})
    scaled = m.scaler.transform(X[-50:])
    probs = m.model.predict_proba(scaled)[:, 1]
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert float(np.std(probs)) > 1e-4, "live probabilities are degenerate/constant"


def test_gauge_contribution_sensible_range(isolated_model_dir):
    """Spot-check: calibrated rare-event probabilities do not pin the ML
    contribution bearish merely because they are below 0.5."""
    df = _make_5min_df(n=900, seed=5)
    m = ShortTrendModel()
    m.train(df, indicators={})
    feats = m.build_latest_features(df, indicators={})
    assert feats is not None
    p = m.predict(feats)
    from signal_engine.short_gauge import ShortTrendGauge

    result = ShortTrendGauge().compute_score(
        {}, p, False, 1.0, "none", neutral_probability=0.10
    )
    ml_term = result["ml_score"]
    assert -40.0 <= ml_term <= 40.0
    assert abs(ml_term) < 40.0
    # the probability itself must be a real number in [0,1]
    assert 0.0 <= p <= 1.0


def test_short_gauge_centers_on_calibrated_base_rate():
    """A normal 10% calibrated event probability is neutral; lower and higher
    probabilities retain directional meaning without changing the score range."""
    from signal_engine.short_gauge import ShortTrendGauge

    gauge = ShortTrendGauge()
    neutral = gauge.compute_score(
        {}, 0.10, False, 1.0, "none", neutral_probability=0.10
    )
    typical = gauge.compute_score(
        {}, 0.08, False, 1.0, "none", neutral_probability=0.10
    )
    bullish = gauge.compute_score(
        {}, 0.25, False, 1.0, "none", neutral_probability=0.10
    )
    assert neutral["ml_score"] == pytest.approx(0.0)
    assert -10.0 < typical["ml_score"] < 0.0
    assert bullish["ml_score"] > 0.0
    for probability in (0.0, 0.08, 0.10, 0.25, 1.0):
        score = gauge.compute_score(
            {}, probability, False, 1.0, "none", neutral_probability=0.10
        )["ml_score"]
        assert -40.0 <= score <= 40.0


# ─────────────────────────────────────────────────────────────────────────────
# Metric-aware regression check plumbing
# ─────────────────────────────────────────────────────────────────────────────

def test_metric_upgrade_transition_is_narrow():
    """Only the legacy→honest upgrade skips the regression check; a fallback
    from honest OOS back to train accuracy must still be regression-checked."""
    from ml.trainer import _is_metric_upgrade_transition

    assert _is_metric_upgrade_transition(None, "purged_walk_forward_oos")
    assert _is_metric_upgrade_transition("train", "purged_walk_forward_oos")
    # fallback / degraded retrain: NOT exempted
    assert not _is_metric_upgrade_transition("purged_walk_forward_oos", "train")
    assert not _is_metric_upgrade_transition("purged_walk_forward_oos", None)
    # steady state: normal comparison
    assert not _is_metric_upgrade_transition(
        "purged_walk_forward_oos", "purged_walk_forward_oos"
    )
    assert not _is_metric_upgrade_transition("train", "train")


def test_training_status_records_accuracy_metric(tmp_path, monkeypatch):
    from ml import training_status as ts

    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")

    # Old-style success without a metric (pre-fix runs)
    ts.record_training_result("short_trend", success=True, accuracy=0.986)
    assert ts.get_last_successful_accuracy("short_trend") == pytest.approx(0.986)
    assert ts.get_last_successful_accuracy_metric("short_trend") is None

    # New-style honest metric
    ts.record_training_result(
        "short_trend", success=True, accuracy=0.55,
        accuracy_metric="purged_walk_forward_oos",
    )
    assert ts.get_last_successful_accuracy("short_trend") == pytest.approx(0.55)
    assert ts.get_last_successful_accuracy_metric("short_trend") == "purged_walk_forward_oos"

    # Metric carried through a failure
    ts.record_training_result("short_trend", success=False, error="boom")
    assert ts.get_last_successful_accuracy("short_trend") == pytest.approx(0.55)
    assert ts.get_last_successful_accuracy_metric("short_trend") == "purged_walk_forward_oos"
