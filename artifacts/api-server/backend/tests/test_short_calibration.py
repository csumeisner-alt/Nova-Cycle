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
  - get_neutral_probability() reflects the loaded positive_rate across several
    different calibration reports
  - gauge ml_score is zero when ml_prediction equals the base rate
  - a changed on-disk report causes the rate to be re-read (no stale value)
  - missing or invalid report metadata falls back safely to 0.5
"""

import json
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from ml import calibration as cal
from ml import short_trend as st
from ml.short_trend import ShortTrendModel, N_FEATURES


@pytest.fixture
def tmp_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def tmp_model_dir_full(tmp_path, monkeypatch):
    """Patch both cal.MODEL_DIR and st.MODEL_PATH so _maybe_reload() never
    tries to read a production model file from the real models/ directory."""
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(st, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(st, "MODEL_PATH", tmp_path / "short_trend_model.pkl")
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


# ─────────────────────────────────────────────────────────────────────────────
# get_neutral_probability(): calibration report → base rate
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("positive_rate", [0.05, 0.08, 0.12, 0.25, 0.35, 0.48])
def test_get_neutral_probability_matches_report(tmp_model_dir_full, positive_rate):
    """get_neutral_probability() returns the exact positive_rate from a valid
    calibration report. Parametrized over realistic rare-event base rates to
    confirm the gauge's neutral point shifts correctly with each retrain."""
    cal.save_calibration_report({"positive_rate": positive_rate}, "short_trend")
    m = ShortTrendModel()
    result = m.get_neutral_probability()
    assert result == pytest.approx(positive_rate, abs=1e-6)


def test_get_neutral_probability_missing_report_returns_half(tmp_model_dir_full):
    """With no calibration report on disk the safe fallback is 0.5."""
    m = ShortTrendModel()
    assert m.get_neutral_probability() == pytest.approx(0.5)


@pytest.mark.parametrize("bad_rate", [None, "abc", 0.0, 1.0, -0.1, 1.5])
def test_get_neutral_probability_invalid_rate_returns_half(tmp_model_dir_full, bad_rate):
    """Invalid or out-of-range positive_rate values (boundary, non-numeric,
    negative, >1) all fall back to 0.5 rather than biasing the gauge."""
    cal.save_calibration_report({"positive_rate": bad_rate}, "short_trend")
    m = ShortTrendModel()
    assert m.get_neutral_probability() == pytest.approx(0.5)


def test_get_neutral_probability_report_without_rate_key_returns_half(tmp_model_dir_full):
    """A report that exists but has no positive_rate key falls back to 0.5."""
    cal.save_calibration_report({"calibrated": True, "oos_accuracy": 0.55}, "short_trend")
    m = ShortTrendModel()
    assert m.get_neutral_probability() == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Stale-rate guard: report replaced on disk → new rate is picked up
# ─────────────────────────────────────────────────────────────────────────────

def test_get_neutral_probability_does_not_use_stale_rate(tmp_model_dir_full):
    """When the calibration report file is overwritten (simulating a retrain),
    the model re-reads the new positive_rate on the next call rather than
    returning the cached value from before the retrain."""
    report_path = cal.calibration_report_path("short_trend")

    # First report
    cal.save_calibration_report({"positive_rate": 0.08}, "short_trend")
    m = ShortTrendModel()
    assert m.get_neutral_probability() == pytest.approx(0.08)

    # Overwrite the report to simulate a retrain with a higher base rate.
    # Sleep briefly so the OS records a later mtime; write via the path directly
    # to guarantee the file object is closed before the next stat() call.
    time.sleep(0.05)
    report_path.write_text(json.dumps({"positive_rate": 0.30}))

    # Must return the updated rate, not the stale 0.08.
    assert m.get_neutral_probability() == pytest.approx(0.30)


def test_get_neutral_probability_cycles_across_multiple_reports(tmp_model_dir_full):
    """Simulate three consecutive retrains with materially different positive
    rates; each call after a file change returns the current rate."""
    report_path = cal.calibration_report_path("short_trend")
    rates = [0.07, 0.18, 0.42]

    m = ShortTrendModel()
    for rate in rates:
        time.sleep(0.05)
        report_path.write_text(json.dumps({"positive_rate": rate}))
        assert m.get_neutral_probability() == pytest.approx(rate, abs=1e-6), (
            f"Expected neutral probability {rate} after report update"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ShortTrendGauge: ml_score is zero when prediction equals the base rate
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("base_rate", [0.05, 0.08, 0.12, 0.25, 0.35, 0.48, 0.50])
def test_gauge_ml_score_zero_when_prediction_equals_base_rate(base_rate):
    """When ml_prediction == neutral_probability the ML contributes 0 to the
    gauge score, so a market behaving at its normal move rate stays neutral
    regardless of what the calibrated base rate is."""
    from signal_engine.short_gauge import ShortTrendGauge

    gauge = ShortTrendGauge()
    result = gauge.compute_score(
        indicators={
            "latest": {
                "rsi": 50.0,
                "stoch_rsi_k": 50.0,
                "stoch_k": 50.0,
                "bb_pct_b": 0.5,
            }
        },
        ml_prediction=base_rate,
        is_extended=False,
        liquidity_score=1.0,
        gap_type="none",
        age_in_minutes=0.0,
        neutral_probability=base_rate,
    )
    assert result["ml_score"] == pytest.approx(0.0, abs=1e-6), (
        f"ml_score should be 0 when prediction equals base_rate={base_rate}"
    )
    assert result["neutral_probability"] == pytest.approx(
        min(0.99, max(0.01, base_rate)), abs=1e-6
    )


def test_gauge_ml_score_direction_above_and_below_base_rate():
    """Predictions above the base rate produce a positive ml_score (bullish);
    predictions below produce a negative ml_score (bearish). This is
    consistent across a low base rate typical for the short-trend label."""
    from signal_engine.short_gauge import ShortTrendGauge

    gauge = ShortTrendGauge()
    base_rate = 0.08  # realistic calibrated positive rate

    def _ml_score(pred):
        return gauge.compute_score(
            indicators={"latest": {"rsi": 50.0, "stoch_rsi_k": 50.0,
                                   "stoch_k": 50.0, "bb_pct_b": 0.5}},
            ml_prediction=pred,
            is_extended=False,
            liquidity_score=1.0,
            gap_type="none",
            age_in_minutes=0.0,
            neutral_probability=base_rate,
        )["ml_score"]

    assert _ml_score(base_rate) == pytest.approx(0.0, abs=1e-6)
    assert _ml_score(base_rate + 0.10) > 0.0, "Above base rate → bullish ml_score"
    assert _ml_score(max(0.01, base_rate - 0.05)) < 0.0, "Below base rate → bearish ml_score"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: model loads report → gauge uses correct neutral probability
# ─────────────────────────────────────────────────────────────────────────────

def test_model_neutral_probability_feeds_gauge_correctly(tmp_model_dir_full, monkeypatch):
    """get_neutral_probability() from a freshly-loaded report produces an ml_score
    of zero when passed as neutral_probability to the gauge — confirms the wiring
    between ShortTrendModel and ShortTrendGauge is bias-free."""
    from signal_engine.short_gauge import ShortTrendGauge

    positive_rate = 0.11
    cal.save_calibration_report({"positive_rate": positive_rate}, "short_trend")

    m = ShortTrendModel()
    neutral = m.get_neutral_probability()
    assert neutral == pytest.approx(positive_rate)

    gauge = ShortTrendGauge()
    result = gauge.compute_score(
        indicators={"latest": {"rsi": 50.0, "stoch_rsi_k": 50.0,
                               "stoch_k": 50.0, "bb_pct_b": 0.5}},
        ml_prediction=neutral,
        is_extended=False,
        liquidity_score=1.0,
        gap_type="none",
        age_in_minutes=0.0,
        neutral_probability=neutral,
    )
    assert result["ml_score"] == pytest.approx(0.0, abs=1e-6)
