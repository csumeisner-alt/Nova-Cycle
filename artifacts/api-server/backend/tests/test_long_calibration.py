"""
Long-trend probability calibration & walk-forward evaluation tests.

Covers:
  - purged walk-forward evaluation mechanics (embargo, fold purging, metrics)
  - calibrator fitting (sigmoid / isotonic selection, degenerate inputs)
  - persistence round-trip (calibrator pickle + JSON report)
  - LongTrendModel.predict applies the calibrator and stays within [0, 1]
  - train() output includes the calibration summary
  - gauge threshold logic is untouched (probabilities in, same score paths)
"""

import json
import pickle

import numpy as np
import pytest

from ml import calibration as cal
from ml.long_trend import LongTrendModel


class DummyModel:
    """Deterministic stand-in classifier for walk-forward tests."""

    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y, sample_weight=None, verbose=False):
        self.fit_calls.append((len(X), len(y)))
        return self

    def predict_proba(self, X):
        # Probability driven by the first feature, squashed into (0, 1)
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - p, p])


def _make_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    # Label correlated with feature 0 → learnable but noisy
    y = ((x[:, 0] + rng.normal(scale=1.5, size=n)) > 0).astype(int)
    w = np.ones(n, dtype=np.float32)
    return x, y, w


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward evaluation
# ─────────────────────────────────────────────────────────────────────────────

def test_walk_forward_reports_metrics_and_pooled_oos():
    X, y, w = _make_series()
    metrics, probs, labels = cal.walk_forward_evaluate(
        X, y, w, model_factory=DummyModel
    )
    assert metrics["evaluated"] is True
    assert metrics["method"] == "purged_walk_forward"
    assert metrics["embargo_rows"] == cal.LABEL_HORIZON
    assert metrics["oos_samples"] == len(probs) == len(labels) > 0
    assert 0.0 <= metrics["oos_accuracy"] <= 1.0
    assert 0.0 <= metrics["oos_brier_score"] <= 1.0
    assert len(metrics["reliability_bins"]) == cal.RELIABILITY_BINS
    # every fold trains strictly before its test window minus the embargo
    for fold in metrics["folds"]:
        assert fold["train_rows"] >= 100


def test_walk_forward_embargo_purges_training_rows():
    """Each fold's training slice must end at least `embargo` rows before
    the test window starts (no label look-ahead)."""
    X, y, w = _make_series()

    seen = []

    class RecordingModel(DummyModel):
        def fit(self, Xf, yf, sample_weight=None, verbose=False):
            seen.append(len(Xf))
            return super().fit(Xf, yf, sample_weight=sample_weight)

    metrics, _, _ = cal.walk_forward_evaluate(X, y, w, model_factory=RecordingModel)
    assert metrics["evaluated"] is True
    # Reconstruct fold boundaries the same way and assert the purge gap.
    n = len(X)
    test_start = max(max(100, cal.LABEL_HORIZON * 3) + cal.LABEL_HORIZON, int(n * 0.5))
    edges = np.linspace(test_start, n, metrics["n_splits"] + 1, dtype=int)
    for train_rows, t0 in zip(seen, edges[:-1]):
        assert train_rows <= int(t0) - cal.LABEL_HORIZON


def test_walk_forward_insufficient_data_is_flagged():
    X, y, w = _make_series(n=60)
    metrics, probs, labels = cal.walk_forward_evaluate(
        X, y, w, model_factory=DummyModel
    )
    assert metrics["evaluated"] is False
    assert len(probs) == 0 and len(labels) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Calibrator fitting
# ─────────────────────────────────────────────────────────────────────────────

def test_fit_calibrator_sigmoid_small_sample():
    rng = np.random.default_rng(1)
    probs = rng.uniform(size=100)
    labels = (probs + rng.normal(scale=0.3, size=100) > 0.5).astype(int)
    c = cal.fit_calibrator(probs, labels)
    assert c is not None
    assert c.method == "sigmoid"
    out = c.transform(0.7)
    assert 0.0 <= out <= 1.0


def test_fit_calibrator_isotonic_large_sample():
    rng = np.random.default_rng(2)
    probs = rng.uniform(size=cal.MIN_ISOTONIC_SAMPLES + 50)
    labels = (probs + rng.normal(scale=0.3, size=len(probs)) > 0.5).astype(int)
    c = cal.fit_calibrator(probs, labels)
    assert c is not None
    assert c.method == "isotonic"
    # isotonic output is monotone in the input
    lo, hi = c.transform(0.1), c.transform(0.9)
    assert lo <= hi


def test_fit_calibrator_rejects_degenerate_inputs():
    assert cal.fit_calibrator(np.array([0.5] * 10), np.array([1] * 10)) is None
    # single-class labels
    probs = np.linspace(0, 1, 200)
    assert cal.fit_calibrator(probs, np.ones(200, dtype=int)) is None


def test_calibrated_brier_improves_or_matches_overconfident_probs():
    """An over-confident predictor's Brier score should not get worse after
    sigmoid calibration on the same sample."""
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=400)
    # Over-confident: pushes weak evidence to extremes
    base = np.where(labels == 1, 0.55, 0.45) + rng.normal(scale=0.05, size=400)
    probs = np.clip(np.where(base > 0.5, base + 0.35, base - 0.35), 0.0, 1.0)
    raw_brier = float(np.mean((probs - labels) ** 2))
    c = cal.fit_calibrator(probs, labels)
    assert c is not None
    cal_brier = cal.calibrated_brier(c, probs, labels)
    assert cal_brier is not None
    assert cal_brier <= raw_brier + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(cal, "CALIBRATOR_PATH", tmp_path / "long_trend_calibrator.pkl")
    monkeypatch.setattr(cal, "REPORT_PATH", tmp_path / "long_trend_calibration.json")
    return tmp_path


def test_calibrator_persistence_roundtrip(tmp_paths):
    rng = np.random.default_rng(4)
    probs = rng.uniform(size=200)
    labels = (probs > 0.5).astype(int)
    labels[:20] = 1 - labels[:20]
    c = cal.fit_calibrator(probs, labels)
    assert cal.save_calibrator(c) is True
    loaded = cal.load_calibrator()
    assert loaded is not None
    assert loaded.method == c.method
    assert loaded.transform(0.6) == pytest.approx(c.transform(0.6))


def test_report_persistence_roundtrip(tmp_paths):
    cal.save_calibration_report({"evaluated": True, "oos_accuracy": 0.55})
    report = cal.get_calibration_report()
    assert report["evaluated"] is True
    assert report["oos_accuracy"] == 0.55
    assert "generated_at" in report


def test_load_calibrator_ignores_foreign_pickle(tmp_paths):
    with open(cal.CALIBRATOR_PATH, "wb") as f:
        pickle.dump({"not": "a calibrator"}, f)
    assert cal.load_calibrator() is None


# ─────────────────────────────────────────────────────────────────────────────
# LongTrendModel integration
# ─────────────────────────────────────────────────────────────────────────────

class ConstantProbModel:
    n_features_in_ = 15

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]] * len(X))


class IdentityCalibrator(cal.ProbabilityCalibrator):
    def __init__(self, offset=-0.3):
        self.method = "test"
        self.offset = offset

    def transform(self, prob):
        return float(np.clip(prob + self.offset, 0.0, 1.0))


def test_predict_applies_calibrator(monkeypatch):
    m = LongTrendModel()
    m.model = ConstantProbModel()
    m._model_loaded = True
    m.calibrator = IdentityCalibrator()
    monkeypatch.setattr(m, "_maybe_reload", lambda: None)
    p = m.predict(np.zeros(15, dtype=np.float32))
    assert p == pytest.approx(0.6)
    assert m.last_prediction_was_fallback is False


def test_predict_without_calibrator_returns_raw(monkeypatch):
    m = LongTrendModel()
    m.model = ConstantProbModel()
    m._model_loaded = True
    m.calibrator = None
    monkeypatch.setattr(m, "_maybe_reload", lambda: None)
    assert m.predict(np.zeros(15, dtype=np.float32)) == pytest.approx(0.9)


def test_predict_survives_broken_calibrator(monkeypatch):
    class Broken:
        def transform(self, prob):
            raise RuntimeError("boom")

    m = LongTrendModel()
    m.model = ConstantProbModel()
    m._model_loaded = True
    m.calibrator = Broken()
    monkeypatch.setattr(m, "_maybe_reload", lambda: None)
    # raw probability is served when the calibrator explodes
    assert m.predict(np.zeros(15, dtype=np.float32)) == pytest.approx(0.9)


def test_train_returns_calibration_summary(tmp_paths, monkeypatch, tmp_path):
    """End-to-end: train on synthetic daily data, expect a calibration block
    in the result and a persisted report + calibrator."""
    import pandas as pd
    from ml import long_trend as lt

    monkeypatch.setattr(lt, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(lt, "MODEL_PATH", tmp_path / "long_trend_model.pkl")

    n = 700
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    drift = np.cumsum(rng.normal(loc=0.03, scale=1.0, size=n))
    close = 400 + drift
    df = pd.DataFrame({
        "open": close - rng.uniform(0, 1, n),
        "high": close + rng.uniform(0, 1, n),
        "low": close - rng.uniform(0, 2, n),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "is_extended_hours": False,
    }, index=idx)

    m = lt.LongTrendModel()
    result = m.train(df, indicators={})
    assert "calibration" in result
    summary = result["calibration"]
    assert isinstance(summary, dict)
    if summary.get("evaluated"):
        assert summary["oos_samples"] > 0
        report = cal.get_calibration_report()
        assert report is not None and report.get("evaluated") is True
        if summary.get("calibrated"):
            assert cal.load_calibrator() is not None
    # thresholds/signal logic untouched: prediction still a [0,1] probability
    feats = m.build_latest_features(df, {})
    assert feats is not None
    p = m.predict(feats)
    assert 0.0 <= p <= 1.0
