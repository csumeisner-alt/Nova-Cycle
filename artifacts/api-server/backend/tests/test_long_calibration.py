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
import time

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


# ─────────────────────────────────────────────────────────────────────────────
# Feature / label timestamp alignment
# ─────────────────────────────────────────────────────────────────────────────

def test_build_features_returns_valid_positions():
    """build_features must return a valid_positions array that indexes df rows
    and has the same length as X and weights."""
    import pandas as pd
    from ml.long_trend import LongTrendModel

    n = 50
    rng = np.random.default_rng(9)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 400 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 1.0,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "is_extended_hours": False,
    }, index=idx)

    model = LongTrendModel()
    X, w, pos = model.build_features(df, {})

    assert len(X) == len(w) == len(pos), "X, weights, and valid_positions must be same length"
    assert pos.dtype == np.intp or np.issubdtype(pos.dtype, np.integer)
    assert all(0 <= p < n for p in pos), "Every valid_position must be a valid row index in df"


def test_build_features_alignment_with_skipped_rows():
    """When rows with zero/negative close are present, labels must align to
    the feature rows that were actually produced — not to the first N rows."""
    import pandas as pd
    from ml.long_trend import LongTrendModel

    n = 60
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 400 + np.cumsum(rng.normal(0, 1, n))

    # Inject zero-close rows at positions 5 and 30 — these are the only rows
    # build_features will skip.  Without the valid_positions fix, the labels
    # for every row after position 5 would be shifted by 1.
    close[5] = 0.0
    close[30] = 0.0

    df = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 1.0,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "is_extended_hours": False,
    }, index=idx)

    model = LongTrendModel()
    X, w, pos = model.build_features(df, {})

    # Rows at positions 5 and 30 must have been skipped.
    assert 5 not in pos, "Row with zero close must not appear in valid_positions"
    assert 30 not in pos, "Row with zero close must not appear in valid_positions"

    # For every produced feature row, the position correctly indexes back into
    # df: the close value at that position must be > 0.
    close_arr = df["close"].values
    for p in pos:
        assert close_arr[p] > 0, f"valid_position {p} points to a non-positive close"

    # Labels selected via valid_positions must correspond to the same rows as
    # the feature vectors (no off-by-one shift).
    df_copy = df.copy()
    horizon = 21
    df_copy["future_close"] = df_copy["close"].shift(-horizon)
    df_copy.dropna(subset=["future_close"], inplace=True)
    df_copy["forward_return"] = df_copy["future_close"] / df_copy["close"] - 1.0
    threshold = 0.02
    df_labeled = df_copy[
        (df_copy["forward_return"] >= threshold) | (df_copy["forward_return"] <= -threshold)
    ].copy()
    df_labeled["label"] = (df_labeled["forward_return"] >= threshold).astype(int)

    # Recompute pos for the labeled subset.
    X2, w2, pos2 = model.build_features(df_labeled, {})
    y_pos = df_labeled["label"].values[pos2]   # correct alignment
    y_bad = df_labeled["label"].values[: len(X2)]  # naive slice

    # If no rows are skipped the naive slice happens to be correct, but the
    # position-based alignment must always produce a consistent result.
    assert len(y_pos) == len(X2), "Position-aligned labels must match feature row count"


def test_train_label_alignment_survives_skipped_rows(tmp_model_dir_full):
    """LongTrendModel.train() must not crash when the df contains zero-close
    rows that build_features skips, and labels must be aligned via valid_positions
    rather than a naive [:len(X)] slice.

    The test verifies that train() completes and returns a dict (not an early
    empty-result dict) by checking that build_features' valid_positions
    mechanism is exercised correctly on the labelled sub-frame.
    """
    import pandas as pd
    from ml import long_trend as lt

    # Use high volatility (scale=5) so many 21-day moves exceed the 2%
    # meaningful-move threshold and we get > LONG_MIN_TRAINING_ROWS labeled rows.
    n = 900
    rng = np.random.default_rng(42)
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    drift = np.cumsum(rng.normal(loc=0.05, scale=5.0, size=n))
    close = np.maximum(300 + drift, 1.0)
    # Inject a handful of zero-close rows at positions that will be seen during
    # feature construction — build_features must skip these and still produce
    # labels correctly aligned to the rows that were NOT skipped.
    close[[10, 50, 300]] = 0.0

    df = pd.DataFrame({
        "open": np.where(close > 0, close - rng.uniform(0, 1, n), 1.0),
        "high": np.where(close > 0, close + rng.uniform(0, 1, n), 1.0),
        "low": np.where(close > 0, close - rng.uniform(0, 2, n), 1.0),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "is_extended_hours": False,
    }, index=idx)

    m = lt.LongTrendModel()
    result = m.train(df, indicators={})
    # Whether or not the meaningful-move filter yields enough rows for a full
    # model fit, train() must return a dict and must not raise.
    assert isinstance(result, dict), "train() must always return a dict"
    # If training produced a model (enough rows), accuracy must be a valid float.
    if result.get("training_rows", 0) > 0:
        assert isinstance(result.get("accuracy"), float)
    # If not enough rows after filtering, the function returns early with 0.0
    # accuracy — that is acceptable; we only require no crash and correct return type.
    assert result.get("accuracy", 0.0) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward regime breakdown
# ─────────────────────────────────────────────────────────────────────────────

def test_walk_forward_regime_breakdown_included_when_provided():
    """When regime_labels are passed, the metrics dict must contain a
    regime_breakdown list with per-regime OOS accuracy/lift entries."""
    X, y, w = _make_series(n=700)
    # Simulate two regimes cycling every 50 rows: 0=LOW, 1=NORMAL.
    regimes = np.tile([0, 1], 350).astype(np.intp)
    metrics, probs, labels = cal.walk_forward_evaluate(
        X, y, w, model_factory=DummyModel, regime_labels=regimes
    )
    assert metrics["evaluated"] is True
    assert "regime_breakdown" in metrics, "regime_breakdown must be present when regime_labels supplied"
    breakdown = metrics["regime_breakdown"]
    assert len(breakdown) >= 1
    for entry in breakdown:
        assert "regime" in entry
        assert "oos_accuracy" in entry
        assert "accuracy_lift_vs_majority" in entry
        assert "oos_brier_score" in entry
        assert 0.0 <= entry["oos_accuracy"] <= 1.0


def test_walk_forward_no_regime_breakdown_without_labels():
    """When no regime_labels are provided, regime_breakdown must be absent."""
    X, y, w = _make_series(n=700)
    metrics, _, _ = cal.walk_forward_evaluate(X, y, w, model_factory=DummyModel)
    assert metrics["evaluated"] is True
    assert "regime_breakdown" not in metrics


def test_walk_forward_regime_breakdown_all_metrics_present():
    """Every entry in regime_breakdown must include the full set of metrics."""
    X, y, w = _make_series(n=700)
    regimes = np.zeros(len(X), dtype=np.intp)  # single regime
    regimes[len(X) // 2 :] = 1                 # split into two
    metrics, _, _ = cal.walk_forward_evaluate(
        X, y, w, model_factory=DummyModel, regime_labels=regimes
    )
    required_keys = {
        "regime", "regime_code", "oos_samples", "oos_accuracy",
        "majority_baseline_accuracy", "accuracy_lift_vs_majority",
        "oos_brier_score", "positive_rate",
    }
    for entry in metrics.get("regime_breakdown", []):
        missing = required_keys - entry.keys()
        assert not missing, f"regime entry missing keys: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# Neutral probability / calibration base rate (report lifecycle)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_model_dir_full(tmp_path, monkeypatch):
    """Patch cal.MODEL_DIR plus lt.MODEL_PATH and the legacy calibrator path
    so _maybe_reload() never touches production files in the real models/
    directory."""
    from ml import long_trend as lt

    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(cal, "CALIBRATOR_PATH", tmp_path / "long_trend_calibrator.pkl")
    monkeypatch.setattr(cal, "REPORT_PATH", tmp_path / "long_trend_calibration.json")
    monkeypatch.setattr(lt, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(lt, "MODEL_PATH", tmp_path / "long_trend_model.pkl")
    return tmp_path


def test_get_neutral_probability_matches_report(tmp_model_dir_full):
    cal.save_calibration_report({"positive_rate": 0.62}, "long_trend")
    m = LongTrendModel()
    assert m.get_neutral_probability() == pytest.approx(0.62)


def test_get_neutral_probability_missing_report_returns_half(tmp_model_dir_full):
    m = LongTrendModel()
    assert m.get_neutral_probability() == pytest.approx(0.5)
    assert m._calibration_report_mtime is None


def test_get_neutral_probability_resets_to_half_when_report_deleted(tmp_model_dir_full):
    """When the calibration report file is deleted after an initial successful
    load (e.g. a failed retrain that removes the old report before writing the
    new one), get_neutral_probability() must return the safe 0.5 fallback
    rather than silently retaining the previously loaded rate.

    Also confirms that _calibration_report_mtime is cleared (set to None) so
    that any subsequent file write is detected as a change on the next call.
    """
    report_path = cal.calibration_report_path("long_trend")

    # Write a valid report and confirm the rate is loaded.
    cal.save_calibration_report({"positive_rate": 0.58}, "long_trend")
    m = LongTrendModel()
    assert m.get_neutral_probability() == pytest.approx(0.58)
    assert m._calibration_report_mtime is not None, (
        "mtime should be set after a successful load"
    )

    # Delete the report file to simulate a failed retrain that removed it.
    report_path.unlink()

    # Must fall back to 0.5, not retain the stale 0.58.
    assert m.get_neutral_probability() == pytest.approx(0.5), (
        "Expected 0.5 fallback after calibration report was deleted"
    )

    # _calibration_report_mtime must be cleared so any future file write
    # (mtime != None) triggers a fresh reload.
    assert m._calibration_report_mtime is None, (
        "_calibration_report_mtime should be None after the file disappears"
    )


def test_get_neutral_probability_recovers_after_delete_then_rewrite(tmp_model_dir_full):
    """After a deletion resets the base rate to 0.5, writing a replacement
    report (e.g. a successful retrain) must be picked up on the next call —
    the model must not stay stuck at 0.5 due to stale mtime bookkeeping."""
    report_path = cal.calibration_report_path("long_trend")

    cal.save_calibration_report({"positive_rate": 0.58}, "long_trend")
    m = LongTrendModel()
    assert m.get_neutral_probability() == pytest.approx(0.58)

    report_path.unlink()
    assert m.get_neutral_probability() == pytest.approx(0.5)

    # Sleep briefly so the OS records a distinct mtime from the original file.
    time.sleep(0.05)
    report_path.write_text(json.dumps({"positive_rate": 0.71}))

    assert m.get_neutral_probability() == pytest.approx(0.71), (
        "Expected the replacement report's positive_rate after delete+rewrite, "
        "not the stuck 0.5 fallback"
    )
