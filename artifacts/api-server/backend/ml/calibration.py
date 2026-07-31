"""
NovaCycle Long-Trend Probability Calibration
============================================
Walk-forward evaluation + probability calibration for the long-trend model.

Why: the long-trend XGBoost model's raw predict_proba output is treated as
"confidence" by the gauge, but its out-of-sample accuracy is barely above
chance, so raw probabilities are badly over-confident. This module:

  1. Runs a purged, chronological walk-forward evaluation (embargo gap at
     least equal to the 21-day label horizon) and reports honest
     out-of-sample accuracy, Brier score, and reliability-curve bins.
  2. Fits a calibrator (isotonic when enough out-of-sample points exist,
     Platt/sigmoid otherwise) on the pooled out-of-sample predictions.
  3. Persists the calibrator (ml/models/long_trend_calibrator.pkl) and the
     evaluation report (ml/models/long_trend_calibration.json) so inference
     and /api/healthz can consume them.

BUY/SELL thresholds and signal logic are untouched — calibration only makes
the probability fed into the existing gauge honest.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
CALIBRATOR_PATH = MODEL_DIR / "long_trend_calibrator.pkl"
REPORT_PATH = MODEL_DIR / "long_trend_calibration.json"


def calibrator_path(model_name: str = "long_trend") -> Path:
    """Per-model calibrator pickle path (ml/models/<name>_calibrator.pkl)."""
    return MODEL_DIR / f"{model_name}_calibrator.pkl"


def calibration_report_path(model_name: str = "long_trend") -> Path:
    """Per-model calibration report path (ml/models/<name>_calibration.json)."""
    return MODEL_DIR / f"{model_name}_calibration.json"

# Label horizon of the long-trend model (21-day forward return). The embargo
# gap between each walk-forward train window and its test window must be at
# least this many rows, or training labels leak information from test prices.
LABEL_HORIZON = 21

# Minimum pooled out-of-sample points before isotonic regression is trusted;
# below this Platt (sigmoid) scaling is used, which is robust on small samples.
MIN_ISOTONIC_SAMPLES = 500

RELIABILITY_BINS = 10


class ProbabilityCalibrator:
    """Picklable wrapper around an isotonic or sigmoid calibration map."""

    def __init__(self, method: str, model):
        self.method = method  # "isotonic" | "sigmoid"
        self._model = model

    def transform(self, prob: float) -> float:
        """Map a raw model probability to a calibrated probability in [0, 1]."""
        p = float(np.clip(prob, 0.0, 1.0))
        if self.method == "isotonic":
            out = float(self._model.predict([p])[0])
        else:  # sigmoid / Platt: logistic regression on the raw probability
            out = float(self._model.predict_proba(np.array([[p]]))[0][1])
        return float(np.clip(out, 0.0, 1.0))


def _reliability_bins(probs: np.ndarray, labels: np.ndarray) -> list:
    """Equal-width reliability-curve bins: mean predicted vs observed rate."""
    bins = []
    edges = np.linspace(0.0, 1.0, RELIABILITY_BINS + 1)
    for i in range(RELIABILITY_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < RELIABILITY_BINS - 1 else (
            (probs >= lo) & (probs <= hi)
        )
        n = int(mask.sum())
        bins.append({
            "bin_low": round(float(lo), 2),
            "bin_high": round(float(hi), 2),
            "count": n,
            "mean_predicted": float(probs[mask].mean()) if n else None,
            "observed_positive_rate": float(labels[mask].mean()) if n else None,
        })
    return bins


def walk_forward_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray],
    model_factory: Callable,
    n_splits: int = 5,
    embargo: int = LABEL_HORIZON,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Purged, chronological walk-forward evaluation.

    The sample is split into `n_splits` sequential test windows over the
    latter part of the series; each fold trains on all rows strictly before
    the test window minus an `embargo` gap (>= label horizon) so no training
    label overlaps test prices.

    Returns:
        (metrics dict, pooled out-of-sample probs, pooled labels)
    """
    n = len(X)
    # Need a meaningful first training window plus test data.
    min_train = max(100, embargo * 3)
    if n < min_train + embargo + n_splits:
        return (
            {"evaluated": False, "reason": f"not enough rows ({n}) for walk-forward"},
            np.array([]),
            np.array([]),
        )

    test_start = max(min_train + embargo, int(n * 0.5))
    fold_edges = np.linspace(test_start, n, n_splits + 1, dtype=int)

    oos_probs: list = []
    oos_labels: list = []
    fold_stats = []

    for k in range(n_splits):
        t0, t1 = int(fold_edges[k]), int(fold_edges[k + 1])
        if t1 <= t0:
            continue
        train_end = t0 - embargo  # purge: drop rows whose label window overlaps
        if train_end < min_train:
            continue
        model = model_factory()
        w = None
        if weights is not None and len(weights) == n:
            w = weights[:train_end].copy()
            mw = float(w.mean())
            if mw > 0:
                w = w / mw
        model.fit(X[:train_end], y[:train_end], sample_weight=w, verbose=False)
        probs = model.predict_proba(X[t0:t1])[:, 1]
        oos_probs.append(probs)
        oos_labels.append(y[t0:t1])
        fold_stats.append({
            "fold": k + 1,
            "train_rows": int(train_end),
            "test_rows": int(t1 - t0),
            "accuracy": float(((probs >= 0.5).astype(int) == y[t0:t1]).mean()),
        })

    if not oos_probs:
        return (
            {"evaluated": False, "reason": "no valid walk-forward folds"},
            np.array([]),
            np.array([]),
        )

    probs = np.concatenate(oos_probs)
    labels = np.concatenate(oos_labels).astype(int)
    accuracy = float(((probs >= 0.5).astype(int) == labels).mean())
    brier = float(np.mean((probs - labels) ** 2))

    metrics = {
        "evaluated": True,
        "method": "purged_walk_forward",
        "n_splits": len(fold_stats),
        "embargo_rows": int(embargo),
        "oos_samples": int(len(probs)),
        "oos_accuracy": accuracy,
        "oos_brier_score": brier,
        "reliability_bins": _reliability_bins(probs, labels),
        "folds": fold_stats,
    }
    return metrics, probs, labels


def fit_calibrator(probs: np.ndarray, labels: np.ndarray) -> Optional[ProbabilityCalibrator]:
    """
    Fit a probability calibrator on pooled out-of-sample predictions.

    Isotonic regression when the sample is large enough to support it,
    otherwise Platt (sigmoid) scaling. Returns None when labels are
    single-class or the sample is degenerate.
    """
    try:
        if len(probs) < 30 or len(np.unique(labels)) < 2:
            return None
        if len(probs) >= MIN_ISOTONIC_SAMPLES:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(probs, labels)
            return ProbabilityCalibrator("isotonic", iso)
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(probs.reshape(-1, 1), labels)
        return ProbabilityCalibrator("sigmoid", lr)
    except Exception as exc:
        logger.error("fit_calibrator error: %s", exc)
        return None


def calibrated_brier(calibrator: ProbabilityCalibrator, probs: np.ndarray,
                     labels: np.ndarray) -> Optional[float]:
    """Brier score of the calibrated probabilities on the same OOS sample."""
    try:
        cal = np.array([calibrator.transform(p) for p in probs])
        return float(np.mean((cal - labels) ** 2))
    except Exception as exc:
        logger.error("calibrated_brier error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_write(path: Path, write_fn) -> None:
    """Write via temp file + fsync + rename so readers never see a partial
    file (predict() reloads the calibrator by mtime; a torn read would
    silently disable calibration until the next retrain)."""
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, write_fn.__mode__) as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_calibrator(calibrator: ProbabilityCalibrator,
                    model_name: str = "long_trend") -> bool:
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        def _write(f):
            pickle.dump(calibrator, f)
        _write.__mode__ = "wb"
        _atomic_write(calibrator_path(model_name), _write)
        return True
    except Exception as exc:
        logger.error("save_calibrator error (%s): %s", model_name, exc)
        return False


def load_calibrator(model_name: str = "long_trend") -> Optional[ProbabilityCalibrator]:
    try:
        path = calibrator_path(model_name)
        if path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, ProbabilityCalibrator):
                return obj
            logger.warning(
                "%s calibrator file has unexpected type; ignoring", model_name
            )
        return None
    except Exception as exc:
        logger.error("load_calibrator error (%s): %s", model_name, exc)
        return None


def save_calibration_report(metrics: dict, model_name: str = "long_trend") -> None:
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(metrics)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()

        def _write(f):
            json.dump(payload, f, indent=2)
        _write.__mode__ = "w"
        _atomic_write(calibration_report_path(model_name), _write)
    except Exception as exc:
        logger.error("save_calibration_report error (%s): %s", model_name, exc)


def _walkforward_report_path(model_name: str) -> Path:
    return MODEL_DIR / f"{model_name}_walkforward.json"


def save_walkforward_report(model_name: str, metrics: dict) -> None:
    """Persist a model's walk-forward evaluation report (never raises)."""
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(metrics)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()

        def _write(f):
            json.dump(payload, f, indent=2)
        _write.__mode__ = "w"
        _atomic_write(_walkforward_report_path(model_name), _write)
    except Exception as exc:
        logger.error("save_walkforward_report error: %s", exc)


def get_walkforward_report(model_name: str) -> Optional[dict]:
    """Load a model's persisted walk-forward report (never raises)."""
    try:
        path = _walkforward_report_path(model_name)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        return None
    except Exception as exc:
        logger.error("get_walkforward_report error: %s", exc)
        return None


def get_calibration_report(model_name: str = "long_trend") -> Optional[dict]:
    """Load the persisted walk-forward calibration report (never raises)."""
    try:
        path = calibration_report_path(model_name)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        return None
    except Exception as exc:
        logger.error("get_calibration_report error: %s", exc)
        return None
