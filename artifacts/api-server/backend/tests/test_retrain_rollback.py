"""Rollback tests: a flagged (regressed/degenerate) retrain must restore the
previous model file on disk, and the models must reload the restored file —
not keep serving the regressed in-memory model — on the next predict.

Covers:
  1. ModelTrainer.run_initial_training rolls the .pkl back to the pre-retrain
     bytes when the accuracy-regression or degeneracy check flags the retrain.
  2. LongTrendModel/ShortTrendModel _maybe_reload() picks up the restored file
     (fresh mtime) and predictions come from the last known-good model.
"""

import asyncio
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from ml import long_trend as lt
from ml import short_trend as st
from ml import training_status as ts
from ml.long_trend import LongTrendModel, FEATURE_NAMES as LONG_FEATURES
from ml.short_trend import ShortTrendModel, N_FEATURES
from ml.trainer import ModelTrainer, _backup_model_file, _restore_model_file


def _daily_df(n=200, seed=42):
    idx = pd.bdate_range("2025-06-02", periods=n)
    rng = np.random.default_rng(seed)
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


def _fivemin_df(n=400, seed=7):
    idx = pd.date_range("2026-07-20 13:30", periods=n, freq="5min")
    rng = np.random.default_rng(seed)
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
def isolated_paths(tmp_path, monkeypatch):
    """Redirect model pickles, calibration files, and training-status JSON
    away from the real ml/models directory.

    Previously only lt/st MODEL_PATH were redirected; the calibration report
    and calibrator pickle were left pointing at the real directory, so every
    LongTrendModel().train() call wrote into ml/models/.  Now we also patch
    the ml.calibration module's MODEL_DIR (and its derived constants) so that
    save_calibration_report, _sidecar_files backups, and all related writes
    land in tmp_path instead.
    """
    from ml import calibration as cal

    long_path = tmp_path / "long_trend_model.pkl"
    short_path = tmp_path / "short_trend_model.pkl"
    status_path = tmp_path / "training_status.json"

    monkeypatch.setattr(lt, "MODEL_PATH", long_path)
    monkeypatch.setattr(lt, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(st, "MODEL_PATH", short_path)
    monkeypatch.setattr(st, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(ts, "STATUS_PATH", status_path)

    # Redirect all calibration writes (reports, calibrator pickles) to tmp_path
    # so no test in this file can touch the real ml/models directory.
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(cal, "CALIBRATOR_PATH", tmp_path / "long_trend_calibrator.pkl")
    monkeypatch.setattr(cal, "REPORT_PATH", tmp_path / "long_trend_calibration.json")

    return long_path, short_path


async def _noop_save_metadata(*args, **kwargs):
    return None


def _empty_df(*args, **kwargs):
    async def _inner(db):
        return pd.DataFrame()
    return _inner


class TestTrainerRollsBackOnFlaggedRetrain:
    """run_initial_training must leave the pre-retrain model bytes on disk
    when the retrain is flagged as regressed or degenerate."""

    def _make_trainer(self, monkeypatch, daily=None, fivemin=None):
        trainer = ModelTrainer()

        async def _load_daily(db):
            return daily if daily is not None else pd.DataFrame()

        async def _load_fivemin(db):
            return fivemin if fivemin is not None else pd.DataFrame()

        async def _load_vix(db):
            return pd.DataFrame()

        async def _load_spx(db):
            return pd.Series(dtype=float)

        monkeypatch.setattr(ModelTrainer, "_load_daily_voo", staticmethod(_load_daily))
        monkeypatch.setattr(ModelTrainer, "_load_fivemin_voo", staticmethod(_load_fivemin))
        monkeypatch.setattr(ModelTrainer, "_load_vix", staticmethod(_load_vix))
        monkeypatch.setattr(ModelTrainer, "_load_spx_close", staticmethod(_load_spx))
        monkeypatch.setattr(
            ModelTrainer, "_save_metadata", staticmethod(_noop_save_metadata)
        )
        return trainer

    def test_regressed_long_retrain_restores_previous_pkl(
        self, isolated_paths, monkeypatch
    ):
        long_path, _ = isolated_paths
        df = _daily_df()

        # Establish a last known-good model + recorded success accuracy.
        LongTrendModel().train(df, {})
        assert long_path.exists()
        good_bytes = long_path.read_bytes()
        ts.record_training_result("long_trend", success=True, accuracy=0.90)

        trainer = self._make_trainer(monkeypatch, daily=df)

        # Simulated regressed retrain: overwrites the pkl and reports an
        # accuracy far below the recorded 0.90 (> MAX_ACCURACY_DROP).
        def _regressed_train(self, d, indicators):
            time.sleep(0.01)
            lt.MODEL_PATH.write_bytes(b"regressed-model-bytes")
            self.model = object()  # non-None so the trainer reaches the regression check
            return {"accuracy": 0.10, "feature_importances": {}, "degenerate": False}

        monkeypatch.setattr(LongTrendModel, "train", _regressed_train)

        asyncio.run(trainer.run_initial_training(object()))

        # Rollback happened: pre-retrain bytes are back on disk.
        assert long_path.read_bytes() == good_bytes
        # And the failure was recorded (so the health surface + shortened
        # retry interval kick in).
        status = ts.get_training_status()["long_trend"]
        assert status["success"] is False
        assert "regression" in (status["error"] or "")

    def test_degenerate_short_retrain_restores_previous_pkl(
        self, isolated_paths, monkeypatch
    ):
        _, short_path = isolated_paths
        daily = _daily_df()
        fivemin = _fivemin_df()

        # Last known-good short model on disk.
        ShortTrendModel().train(fivemin, {})
        assert short_path.exists()
        good_bytes = short_path.read_bytes()
        ts.record_training_result("short_trend", success=True, accuracy=0.80)

        trainer = self._make_trainer(monkeypatch, daily=daily, fivemin=fivemin)

        # Long-trend retrain succeeds without touching anything.
        def _long_ok(self, d, indicators):
            self.model = object()
            return {"accuracy": 0.90, "feature_importances": {}, "degenerate": False}

        # Short-trend retrain writes a broken pkl and is flagged degenerate.
        def _degenerate_train(self, d, indicators):
            time.sleep(0.01)
            st.MODEL_PATH.write_bytes(b"degenerate-model-bytes")
            self.model = object()
            return {
                "accuracy": 0.55,
                "val_accuracy": 0.55,
                "degenerate": True,
                "degeneracy_reason": "constant predictions",
            }

        monkeypatch.setattr(LongTrendModel, "train", _long_ok)
        monkeypatch.setattr(ShortTrendModel, "train", _degenerate_train)

        asyncio.run(trainer.run_initial_training(object()))

        assert short_path.read_bytes() == good_bytes
        status = ts.get_training_status()["short_trend"]
        assert status["success"] is False
        assert "Degenerate" in (status["error"] or "")


class TestFlaggedRetrainSkipsMetadata:
    """A flagged (regressed/degenerate) retrain must NOT persist a
    ModelMetadata row with the discarded accuracy — health endpoints read the
    latest row and must keep showing the restored last-good model."""

    def _make_trainer(self, monkeypatch, saved, daily=None, fivemin=None):
        trainer = ModelTrainer()

        async def _load_daily(db):
            return daily if daily is not None else pd.DataFrame()

        async def _load_fivemin(db):
            return fivemin if fivemin is not None else pd.DataFrame()

        async def _load_vix(db):
            return pd.DataFrame()

        async def _load_spx(db):
            return pd.Series(dtype=float)

        async def _record_save_metadata(db, model_name, ticker, accuracy, feature_importances):
            saved.append({"model_name": model_name, "accuracy": accuracy})

        monkeypatch.setattr(ModelTrainer, "_load_daily_voo", staticmethod(_load_daily))
        monkeypatch.setattr(ModelTrainer, "_load_fivemin_voo", staticmethod(_load_fivemin))
        monkeypatch.setattr(ModelTrainer, "_load_vix", staticmethod(_load_vix))
        monkeypatch.setattr(ModelTrainer, "_load_spx_close", staticmethod(_load_spx))
        monkeypatch.setattr(
            ModelTrainer, "_save_metadata", staticmethod(_record_save_metadata)
        )
        return trainer

    def test_regressed_long_retrain_writes_no_metadata_row(
        self, isolated_paths, monkeypatch
    ):
        long_path, _ = isolated_paths
        df = _daily_df()

        LongTrendModel().train(df, {})
        ts.record_training_result("long_trend", success=True, accuracy=0.90)

        saved = []
        trainer = self._make_trainer(monkeypatch, saved, daily=df)

        def _regressed_train(self, d, indicators):
            lt.MODEL_PATH.write_bytes(b"regressed-model-bytes")
            self.model = object()
            return {"accuracy": 0.10, "feature_importances": {}, "degenerate": False}

        monkeypatch.setattr(LongTrendModel, "train", _regressed_train)

        asyncio.run(trainer.run_initial_training(object()))

        # No metadata row written for the flagged long retrain.
        assert all(rec["model_name"] != "long_trend" for rec in saved)

    def test_degenerate_short_retrain_writes_no_metadata_but_good_long_does(
        self, isolated_paths, monkeypatch
    ):
        _, short_path = isolated_paths
        daily = _daily_df()
        fivemin = _fivemin_df()

        ShortTrendModel().train(fivemin, {})
        ts.record_training_result("short_trend", success=True, accuracy=0.80)

        saved = []
        trainer = self._make_trainer(monkeypatch, saved, daily=daily, fivemin=fivemin)

        def _long_ok(self, d, indicators):
            self.model = object()
            return {"accuracy": 0.90, "feature_importances": {}, "degenerate": False}

        def _degenerate_train(self, d, indicators):
            st.MODEL_PATH.write_bytes(b"degenerate-model-bytes")
            self.model = object()
            return {
                "accuracy": 0.55,
                "val_accuracy": 0.55,
                "degenerate": True,
                "degeneracy_reason": "constant predictions",
            }

        monkeypatch.setattr(LongTrendModel, "train", _long_ok)
        monkeypatch.setattr(ShortTrendModel, "train", _degenerate_train)

        asyncio.run(trainer.run_initial_training(object()))

        # Successful long retrain persisted its metadata with the new accuracy…
        long_rows = [r for r in saved if r["model_name"] == "long_trend"]
        assert len(long_rows) == 1
        assert long_rows[0]["accuracy"] == pytest.approx(0.90)
        # …but the flagged short retrain wrote no row at all.
        assert all(r["model_name"] != "short_trend" for r in saved)

    def test_successful_retrain_still_writes_metadata(
        self, isolated_paths, monkeypatch
    ):
        df = _daily_df()
        ts.record_training_result("long_trend", success=True, accuracy=0.50)

        saved = []
        trainer = self._make_trainer(monkeypatch, saved, daily=df)

        def _good_train(self, d, indicators):
            self.model = object()
            return {"accuracy": 0.60, "feature_importances": {"rsi": 1.0}, "degenerate": False}

        monkeypatch.setattr(LongTrendModel, "train", _good_train)

        asyncio.run(trainer.run_initial_training(object()))

        long_rows = [r for r in saved if r["model_name"] == "long_trend"]
        assert len(long_rows) == 1
        assert long_rows[0]["accuracy"] == pytest.approx(0.60)


class TestOOSLiftGateRejectsNegativeLiftRetrain:
    """A long-trend retrain whose purged OOS accuracy_lift_vs_majority is at
    or below LONG_MIN_OOS_ACCURACY_LIFT must be rejected and rolled back.
    The scalar ml_confidence API contract (predict → float in [0,1]) is
    verified to remain unchanged by the gate."""

    def _make_trainer(self, monkeypatch, daily=None):
        trainer = ModelTrainer()

        async def _load_daily(db):
            return daily if daily is not None else pd.DataFrame()

        async def _load_fivemin(db):
            return pd.DataFrame()

        async def _load_vix(db):
            return pd.DataFrame()

        async def _load_spx(db):
            return pd.Series(dtype=float)

        async def _noop_meta(*a, **k):
            return None

        monkeypatch.setattr(ModelTrainer, "_load_daily_voo", staticmethod(_load_daily))
        monkeypatch.setattr(ModelTrainer, "_load_fivemin_voo", staticmethod(_load_fivemin))
        monkeypatch.setattr(ModelTrainer, "_load_vix", staticmethod(_load_vix))
        monkeypatch.setattr(ModelTrainer, "_load_spx_close", staticmethod(_load_spx))
        monkeypatch.setattr(
            ModelTrainer, "_save_metadata", staticmethod(_noop_meta)
        )
        return trainer

    def test_negative_oos_lift_rolls_back_model(self, isolated_paths, monkeypatch):
        """A retrain that reports a negative OOS lift (below the majority
        baseline) must be rejected and the pre-retrain model restored."""
        long_path, _ = isolated_paths
        df = _daily_df()

        # Establish a working model file on disk.
        LongTrendModel().train(df, {})
        assert long_path.exists()
        good_bytes = long_path.read_bytes()
        ts.record_training_result("long_trend", success=True, accuracy=0.60)

        trainer = self._make_trainer(monkeypatch, daily=df)

        # Simulate a retrain that writes a new pkl and reports a negative OOS
        # lift — the candidate beats neither the majority baseline nor the
        # existing model.  The gate must detect this and roll back.
        def _below_baseline_train(self_m, d, indicators):
            lt.MODEL_PATH.write_bytes(b"below-baseline-candidate")
            self_m.model = object()  # non-None so gate logic is reached
            return {
                "accuracy": 0.45,  # headline OOS acc
                "accuracy_metric": "purged_walk_forward_oos",
                "feature_importances": {},
                "degenerate": False,
                "calibration": {
                    "evaluated": True,
                    "oos_accuracy": 0.45,
                    "majority_baseline_accuracy": 0.60,
                    "accuracy_lift_vs_majority": -0.15,  # negative lift → reject
                },
            }

        monkeypatch.setattr(LongTrendModel, "train", _below_baseline_train)

        asyncio.run(trainer.run_initial_training(object()))

        # The gate must restore the pre-retrain model bytes.
        assert long_path.read_bytes() == good_bytes, (
            "Pre-retrain model must be restored when OOS lift is negative"
        )

        # The failure must be recorded so health endpoints and the shortened
        # retry interval kick in.
        status = ts.get_training_status().get("long_trend", {})
        assert status.get("success") is False, "Expected failure recorded for rejected retrain"
        assert "OOS quality gate" in (status.get("error") or ""), (
            f"Expected OOS quality gate message in error, got: {status.get('error')}"
        )

    def test_positive_oos_lift_accepts_model(self, isolated_paths, monkeypatch):
        """A retrain with positive OOS lift must be accepted and the model
        on disk replaced.  The ml_confidence scalar contract is unchanged:
        predict() returns a float in [0, 1] after acceptance."""
        long_path, _ = isolated_paths
        df = _daily_df()

        LongTrendModel().train(df, {})
        ts.record_training_result("long_trend", success=True, accuracy=0.50)

        trainer = self._make_trainer(monkeypatch, daily=df)

        new_candidate_bytes = b"accepted-candidate-model"

        def _above_baseline_train(self_m, d, indicators):
            lt.MODEL_PATH.write_bytes(new_candidate_bytes)
            self_m.model = object()
            return {
                "accuracy": 0.65,
                "accuracy_metric": "purged_walk_forward_oos",
                "feature_importances": {},
                "degenerate": False,
                "calibration": {
                    "evaluated": True,
                    "oos_accuracy": 0.65,
                    "majority_baseline_accuracy": 0.55,
                    "accuracy_lift_vs_majority": 0.10,  # positive → accept
                },
            }

        monkeypatch.setattr(LongTrendModel, "train", _above_baseline_train)

        asyncio.run(trainer.run_initial_training(object()))

        # Accepted: the new candidate bytes must be on disk.
        assert long_path.read_bytes() == new_candidate_bytes, (
            "New model must be kept when OOS lift is positive"
        )

        status = ts.get_training_status().get("long_trend", {})
        assert status.get("success") is True, "Expected success recorded for accepted retrain"

    def test_ml_confidence_contract_unchanged(self, isolated_paths, monkeypatch):
        """After a successful retrain the predict() contract (scalar float in
        [0, 1]) must be unchanged — no APK rebuild required."""
        long_path, _ = isolated_paths
        df = _daily_df()

        model = LongTrendModel()
        model.train(df, {})
        X, _, _ = model.build_features(df, {})
        if len(X) == 0:
            pytest.skip("No feature rows produced — insufficient data")

        result = model.predict(X[-1:])
        assert isinstance(result, float), f"predict() must return float, got {type(result)}"
        assert 0.0 <= result <= 1.0, f"predict() must return [0,1], got {result}"


class TestPredictReloadsRestoredModel:
    """After a rollback, the next predict() must serve the restored on-disk
    model (via mtime-based _maybe_reload), not the regressed in-memory one."""

    def test_long_trend_reloads_restored_model(self, isolated_paths):
        long_path, _ = isolated_paths
        df_good = _daily_df(seed=42)
        df_bad = _daily_df(seed=123)

        model = LongTrendModel()
        model.train(df_good, {})
        good_bytes = long_path.read_bytes()

        X, _, _ = model.build_features(df_good, {})
        x = X[-1:]
        good_pred = model.predict(x)

        # Backup (as the trainer does), then a "regressed" retrain replaces
        # both the on-disk pkl and the in-memory model.
        backup = _backup_model_file(long_path)
        time.sleep(0.02)
        model.train(df_bad, {})
        regressed_model = model.model
        regressed_pred = float(regressed_model.predict_proba(x)[0][1])
        assert abs(regressed_pred - good_pred) > 1e-9  # models actually differ

        # Rollback restores the good pkl with a fresh mtime.
        time.sleep(0.02)
        assert _restore_model_file(long_path, backup, "long_trend") is True
        assert long_path.read_bytes() == good_bytes

        # Next predict reloads the restored file, not the in-memory model.
        pred_after_rollback = model.predict(x)
        assert pred_after_rollback == pytest.approx(good_pred, abs=1e-9)
        assert model.model is not regressed_model

    def test_short_trend_reloads_restored_model(self, isolated_paths):
        _, short_path = isolated_paths
        df_good = _fivemin_df(seed=7)
        df_bad = _fivemin_df(seed=99)

        model = ShortTrendModel()
        model.train(df_good, {})
        good_bytes = short_path.read_bytes()

        X, _, _ = model.build_features(df_good, {})
        x = X[-1:]
        good_pred = model.predict(x)

        backup = _backup_model_file(short_path)
        time.sleep(0.02)
        model.train(df_bad, {})
        regressed_model = model.model
        scaled = model.scaler.transform(x)
        regressed_pred = float(regressed_model.predict_proba(scaled)[0][1])
        assert abs(regressed_pred - good_pred) > 1e-9

        time.sleep(0.02)
        assert _restore_model_file(short_path, backup, "short_trend") is True
        assert short_path.read_bytes() == good_bytes

        pred_after_rollback = model.predict(x)
        assert pred_after_rollback == pytest.approx(good_pred, abs=1e-9)
        assert model.model is not regressed_model
