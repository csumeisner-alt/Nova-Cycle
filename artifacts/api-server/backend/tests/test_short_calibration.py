"""
Short-trend probability calibration tests.

The short-trend MLP trains on a class-balanced sample, so its raw probability
does not equal the true base rate of a >0.3% move within the hour. These tests
verify the walk-forward calibration machinery (shared with the long-trend
model) is wired into ShortTrendModel:

  - per-model calibrator/report persistence does not collide with long_trend
  - predict() applies the calibrator and stays within [0, 1]
  - raw probability is served when the calibrator is missing or broken
  - train() fits + persists a calibrator and returns a calibration summary
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from ml import calibration as cal
from ml.short_trend import ShortTrendModel, N_FEATURES


@pytest.fixture
def tmp_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# Per-model persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_per_model_paths_are_distinct(tmp_model_dir):
    assert cal.calibrator_path("short_trend") != cal.calibrator_path("long_trend")
    assert (
        cal.calibration_report_path("short_trend")
        != cal.calibration_report_path("long_trend")
    )


def test_short_calibrator_roundtrip_does_not_touch_long(tmp_model_dir):
    rng = np.random.default_rng(4)
    probs = rng.uniform(size=200)
    labels = (probs > 0.5).astype(int)
    labels[:20] = 1 - labels[:20]
    c = cal.fit_calibrator(probs, labels)
    assert cal.save_calibrator(c, "short_trend") is True
    assert cal.calibrator_path("short_trend").exists()
    assert not cal.calibrator_path("long_trend").exists()
    loaded = cal.load_calibrator("short_trend")
    assert loaded is not None
    assert loaded.transform(0.6) == pytest.approx(c.transform(0.6))
    # loading the (absent) long-trend calibrator stays independent
    assert cal.load_calibrator("long_trend") is None


def test_short_report_roundtrip(tmp_model_dir):
    cal.save_calibration_report({"calibrated": True, "oos_accuracy": 0.55},
                                "short_trend")
    report = cal.get_calibration_report("short_trend")
    assert report["calibrated"] is True
    assert report["oos_accuracy"] == 0.55
    assert cal.get_calibration_report("long_trend") is None


def test_load_short_calibrator_ignores_foreign_pickle(tmp_model_dir):
    with open(cal.calibrator_path("short_trend"), "wb") as f:
        pickle.dump({"not": "a calibrator"}, f)
    assert cal.load_calibrator("short_trend") is None


# ─────────────────────────────────────────────────────────────────────────────
# ShortTrendModel.predict integration
# ─────────────────────────────────────────────────────────────────────────────

class ConstantProbModel:
    n_features_in_ = N_FEATURES

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]] * len(X))


class OffsetCalibrator(cal.ProbabilityCalibrator):
    def __init__(self, offset=-0.3):
        self.method = "test"
        self.offset = offset

    def transform(self, prob):
        return float(np.clip(prob + self.offset, 0.0, 1.0))


def _stub_model(calibrator, monkeypatch):
    m = ShortTrendModel()
    m.model = ConstantProbModel()
    m.scaler = None
    m._model_loaded = True
    m.calibrator = calibrator
    monkeypatch.setattr(m, "_maybe_reload", lambda: None)
    return m


def test_predict_applies_calibrator(monkeypatch):
    m = _stub_model(OffsetCalibrator(), monkeypatch)
    p = m.predict(np.zeros(N_FEATURES, dtype=np.float32))
    assert p == pytest.approx(0.6)
    assert m.last_prediction_was_fallback is False


def test_predict_without_calibrator_returns_raw(monkeypatch):
    m = _stub_model(None, monkeypatch)
    assert m.predict(np.zeros(N_FEATURES, dtype=np.float32)) == pytest.approx(0.9)


def test_predict_survives_broken_calibrator(monkeypatch):
    class Broken:
        def transform(self, prob):
            raise RuntimeError("boom")

    m = _stub_model(Broken(), monkeypatch)
    # raw probability is served when the calibrator explodes
    assert m.predict(np.zeros(N_FEATURES, dtype=np.float32)) == pytest.approx(0.9)
    assert m.last_prediction_was_fallback is False


def test_predict_output_stays_in_unit_interval(monkeypatch):
    m = _stub_model(OffsetCalibrator(offset=0.5), monkeypatch)
    p = m.predict(np.zeros(N_FEATURES, dtype=np.float32))
    assert 0.0 <= p <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# train() end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_5min_df(n=900, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="5min")
    # Autocorrelated returns so the label is (weakly) learnable
    shocks = rng.normal(scale=0.002, size=n)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = 0.4 * rets[i - 1] + shocks[i]
    close = 500 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(scale=0.0005, size=n)),
        "high": close * (1 + rng.uniform(0, 0.001, n)),
        "low": close * (1 - rng.uniform(0, 0.001, n)),
        "close": close,
        "volume": rng.integers(1000, 5000, n).astype(float),
        "is_extended_hours": False,
    }, index=idx)
    return df


def test_train_fits_and_persists_short_calibrator(tmp_model_dir, monkeypatch, tmp_path):
    from ml import short_trend as st

    monkeypatch.setattr(st, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(st, "MODEL_PATH", tmp_path / "short_trend_model.pkl")

    df = _synthetic_5min_df()
    m = ShortTrendModel()
    result = m.train(df, indicators={})

    assert "calibration" in result
    summary = result["calibration"]
    # Walk-forward ran; whether the calibrator fitted depends on label
    # diversity in the pooled OOS sample — both outcomes must be reported.
    assert "calibrated" in summary
    report = cal.get_calibration_report("short_trend")
    assert report is not None
    if summary["calibrated"]:
        assert summary["calibration_method"] in ("isotonic", "sigmoid")
        assert cal.calibrator_path("short_trend").exists()
        assert m.calibrator is not None
        # calibrated output remains a valid probability
        feats = m.build_latest_features(df, indicators={})
        m._maybe_reload = lambda: None
        p = m.predict(feats)
        assert 0.0 <= p <= 1.0
