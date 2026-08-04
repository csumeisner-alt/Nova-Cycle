"""Regression tests for the weekly retrain → save → reload → predict cycle.

Task context: the feature sets of both models were extended and a stale-model
guard discards old pickles (predictions fall back to neutral 0.5 until the
weekly retrain). These tests confirm the full train/save/load/predict
round-trip produces valid (0–1, non-constant) predictions with the extended
feature counts, on synthetic data.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from ml import long_trend as lt
from ml import short_trend as st
from ml.long_trend import LongTrendModel, FEATURE_NAMES as LONG_FEATURES
from ml.short_trend import ShortTrendModel, FEATURE_NAMES as SHORT_FEATURES, N_FEATURES


def _daily_df(n=200):
    idx = pd.bdate_range("2025-06-02", periods=n)
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.normal(0.05, 1.0, n)), index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(100.0),
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
        "is_extended_hours": False,
        "session_type": "regular",
    }, index=idx)


def _fivemin_df(n=400):
    idx = pd.date_range("2026-07-20 13:30", periods=n, freq="5min")
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.15, n)), index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(100.0),
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": rng.uniform(1e4, 5e4, n),
        "is_extended_hours": False,
        "session_type": "regular",
        "gap_percent": 0.5,
        "gap_type": "none",
    }, index=idx)


@pytest.fixture
def isolated_model_paths(tmp_path, monkeypatch):
    """Redirect both models' pickle paths and calibration files so tests
    never touch the real ml/models directory."""
    from ml import calibration as cal

    long_path = tmp_path / "long_trend_model.pkl"
    short_path = tmp_path / "short_trend_model.pkl"
    monkeypatch.setattr(lt, "MODEL_PATH", long_path)
    monkeypatch.setattr(lt, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(st, "MODEL_PATH", short_path)
    monkeypatch.setattr(st, "MODEL_DIR", tmp_path)

    # Redirect calibration report / calibrator writes away from real ml/models/
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(cal, "CALIBRATOR_PATH", tmp_path / "long_trend_calibrator.pkl")
    monkeypatch.setattr(cal, "REPORT_PATH", tmp_path / "long_trend_calibration.json")

    return long_path, short_path


class TestRetrainRoundTrip:
    """Equivalent of ModelTrainer.run_initial_training: train → save →
    reload in a fresh instance → predict, for both models."""

    def test_long_trend_round_trip(self, isolated_model_paths):
        long_path, _ = isolated_model_paths
        df = _daily_df()

        result = LongTrendModel().train(df, {})
        assert result["accuracy"] > 0.0
        assert set(result["feature_importances"].keys()) == set(LONG_FEATURES)
        assert long_path.exists()

        # Fresh instance: load from disk and predict
        fresh = LongTrendModel()
        assert fresh.load_model() is True
        assert int(fresh.model.n_features_in_) == len(LONG_FEATURES)

        X, _w, _pos = fresh.build_features(df, {})
        assert X.shape[1] == len(LONG_FEATURES)
        preds = [fresh.predict(X[i]) for i in range(0, len(X), 10)]
        assert all(0.0 <= p <= 1.0 for p in preds)
        # Non-constant: not the neutral-0.5 fallback, and varies across inputs
        assert np.std(preds) > 1e-6
        assert any(abs(p - 0.5) > 1e-6 for p in preds)

    def test_short_trend_round_trip(self, isolated_model_paths):
        _, short_path = isolated_model_paths
        df = _fivemin_df()

        result = ShortTrendModel().train(df, {})
        assert result["accuracy"] > 0.0
        assert short_path.exists()

        fresh = ShortTrendModel()
        assert fresh.load_model() is True
        assert int(fresh.scaler.n_features_in_) == N_FEATURES == len(SHORT_FEATURES)

        X, _w, _pos = fresh.build_features(df, {})
        assert X.shape[1] == N_FEATURES
        preds = [fresh.predict(X[i]) for i in range(0, len(X), 20)]
        assert all(0.0 <= p <= 1.0 for p in preds)
        assert np.std(preds) > 1e-6
        assert any(abs(p - 0.5) > 1e-6 for p in preds)


class TestLongTrendStaleModelDiscard:
    def test_stale_model_discarded(self, tmp_path, monkeypatch):
        """A long-trend pickle with the old feature count must be discarded,
        not crash — mirroring the short-trend stale-model test."""
        from sklearn.linear_model import LogisticRegression

        old_n = len(LONG_FEATURES) - 4
        stale = LogisticRegression().fit(
            np.vstack([np.zeros((5, old_n)), np.ones((5, old_n))]),
            [0] * 5 + [1] * 5,
        )
        path = tmp_path / "long_trend_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(stale, f)
        monkeypatch.setattr(lt, "MODEL_PATH", path)

        m = LongTrendModel()
        assert m.load_model() is False
        assert m.model is None
        assert m.predict(np.zeros(len(LONG_FEATURES))) == 0.5
