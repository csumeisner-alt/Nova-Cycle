"""Process-boundary fire drills for long-model promotion and rollback.

These tests deliberately run the real ModelTrainer orchestration while using a
small deterministic model writer in place of expensive XGBoost training. Each
target family gets a successful promotion followed by a rejected candidate,
then a fresh LongTrendModel proves that the restored artifact bundle is what
would be served after restart.
"""

import asyncio
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml import calibration as cal
from ml import long_trend as lt
from ml import training_status as status
from ml.long_trend import LongTrendModel
from ml.short_trend import ShortTrendModel
from ml.trainer import ModelTrainer


class _FireDrillModel:
    """Small picklable model with target-aware, deterministic probabilities."""

    def __init__(self, target_type: str, marker: str):
        self.target_type = target_type
        self.marker = marker

    def predict_proba(self, features):
        n = len(features)
        if self.target_type == "three_state":
            if self.marker == "good":
                row = [0.15, 0.25, 0.60]
            else:
                row = [0.60, 0.25, 0.15]
        elif self.marker == "good":
            row = [0.30, 0.70]
        else:
            row = [0.80, 0.20]
        return np.array([row] * n, dtype=float)


class _ConstantCalibrationMap:
    def predict_proba(self, features):
        return np.array([[0.25, 0.75]] * len(features), dtype=float)


def _daily_df():
    index = pd.bdate_range("2025-06-02", periods=8)
    close = pd.Series(np.linspace(100.0, 107.0, len(index)), index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000.0,
            "is_extended_hours": False,
            "session_type": "regular",
        },
        index=index,
    )


@pytest.fixture
def isolated_fire_drill(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(lt, "MODEL_PATH", tmp_path / "long_trend_model.pkl")
    monkeypatch.setattr(lt, "_META_PATH", tmp_path / "long_trend_meta.json")
    monkeypatch.setattr(status, "STATUS_PATH", tmp_path / "training_status.json")
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        cal, "CALIBRATOR_PATH", tmp_path / "long_trend_calibrator.pkl"
    )
    monkeypatch.setattr(cal, "REPORT_PATH", tmp_path / "long_trend_calibration.json")

    async def load_daily(_db):
        return _daily_df()

    async def load_vix(_db):
        return pd.DataFrame()

    async def load_spx(_db):
        return pd.Series(dtype=float)

    async def load_context(_db):
        return pd.Series(dtype=float)

    async def load_fivemin(_db):
        return pd.DataFrame()

    async def no_metadata(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ModelTrainer, "_load_daily_voo", staticmethod(load_daily))
    monkeypatch.setattr(ModelTrainer, "_load_vix", staticmethod(load_vix))
    monkeypatch.setattr(ModelTrainer, "_load_spx_close", staticmethod(load_spx))
    monkeypatch.setattr(
        ModelTrainer, "_load_broader_context", staticmethod(load_context)
    )
    monkeypatch.setattr(ModelTrainer, "_load_fivemin_voo", staticmethod(load_fivemin))
    monkeypatch.setattr(ModelTrainer, "_save_metadata", staticmethod(no_metadata))
    monkeypatch.setattr(
        "ml.post_retrain_ablation.run_broader_context_ablation",
        lambda *_args: {},
    )
    return tmp_path


def _write_candidate(target_type: str, marker: str):
    model = _FireDrillModel(target_type, marker)
    with open(lt.MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    # Use a valid calibrator object so a restarted model can load and apply it,
    # rather than merely checking that an arbitrary file exists.
    calibrator = cal.ProbabilityCalibrator(
        "sigmoid", _ConstantCalibrationMap()
    )
    assert cal.save_calibrator(calibrator, "long_trend") is True
    cal.save_calibration_report(
        {
            "evaluated": True,
            "oos_accuracy": 0.70,
            "positive_rate": 0.55,
            "accuracy_lift_vs_majority": 0.20,
            "precision_lift_vs_base_rate": 2.5,
            "pr_auc": 0.80,
        },
        "long_trend",
        dataset_meta={"total_candles": 8, "labeled_rows": 8},
    )


def _candidate_result(target_type: str, marker: str):
    metadata = lt.build_promotion_meta(target_type)
    if marker == "good":
        calibration = {
            "evaluated": True,
            "oos_accuracy": 0.70,
            "accuracy_lift_vs_majority": 0.20,
        }
        accuracy_metric = (
            "purged_walk_forward_multiclass"
            if target_type == "three_state"
            else "purged_walk_forward_oos"
        )
        result = {
            "accuracy": 0.70,
            "accuracy_metric": accuracy_metric,
            "calibration": calibration,
            "feature_importances": {},
            "degenerate": False,
        }
        if target_type == "three_state":
            calibration.update(
                {
                    "macro_f1": 0.60,
                    "per_class": [
                        {"class": 0, "f1": 0.50},
                        {"class": 1, "f1": 0.55},
                        {"class": 2, "f1": 0.65},
                    ],
                }
            )
            result["macro_f1"] = 0.60
            result["per_class"] = calibration["per_class"]
        elif target_type == "drawdown_event":
            result["accuracy_metric"] = "purged_walk_forward_oos"
            result["pr_auc_lift_vs_prevalence"] = 2.5
            calibration["precision_lift_vs_base_rate"] = 2.5
    else:
        result = {
            "accuracy": 0.0,
            "accuracy_metric": "walk_forward_failed",
            "calibration": {
                "evaluated": False,
                "reason": "fire-drill rejected candidate",
            },
            "feature_importances": {},
            "degenerate": False,
        }

    result.update(
        {
            "target_type": target_type,
            "target_horizon_days": metadata["target_horizon_days"],
            "target_threshold": metadata["target_threshold"],
            "feature_names": metadata["feature_names"],
            "broader_context_enabled": metadata["broader_context_enabled"],
        }
    )
    return result


@pytest.mark.parametrize("target_type", ["direction", "drawdown_event", "three_state"])
def test_long_retrain_promotion_restart_and_rollback_fire_drill(
    isolated_fire_drill, monkeypatch, target_type
):
    """A promoted model survives restart; a rejected candidate cannot leak."""
    monkeypatch.setattr(lt.settings, "LONG_TARGET_TYPE", target_type)
    phase = {"marker": "good"}

    def deterministic_train(self, _daily, _indicators):
        _write_candidate(target_type, phase["marker"])
        self.model = _FireDrillModel(target_type, phase["marker"])
        return _candidate_result(target_type, phase["marker"])

    monkeypatch.setattr(LongTrendModel, "train", deterministic_train)
    monkeypatch.setattr(
        ShortTrendModel,
        "train",
        lambda self, *_args: {"accuracy": 0.6, "degenerate": False},
    )

    trainer = ModelTrainer()
    asyncio.run(trainer.run_initial_training(object()))

    model_path = lt.MODEL_PATH
    calibrator_path = cal.calibrator_path("long_trend")
    report_path = cal.calibration_report_path("long_trend")
    metadata_path = lt._META_PATH
    good_bundle = {
        path.name: path.read_bytes()
        for path in (model_path, calibrator_path, report_path, metadata_path)
    }
    assert set(good_bundle) == {
        "long_trend_model.pkl",
        "long_trend_calibrator.pkl",
        "long_trend_calibration.json",
        "long_trend_meta.json",
    }

    good_status = status.get_training_status()["long_trend"]
    assert good_status["success"] is True
    assert good_status["accuracy_metric"].startswith("purged_walk_forward")

    restarted_good = LongTrendModel()
    assert restarted_good.load_model() is True
    assert restarted_good.is_baseline_mode() is False
    assert restarted_good.target_type == target_type
    good_features = np.zeros(len(lt.current_feature_names()), dtype=np.float32)
    good_prediction = restarted_good.predict(good_features)
    assert restarted_good.last_prediction_was_fallback is False
    assert 0.0 <= good_prediction <= 1.0
    assert json.loads(metadata_path.read_text())["target_type"] == target_type

    # Candidate training writes every artifact, but its failed OOS evidence
    # must cause the real trainer to restore the complete previous bundle.
    phase["marker"] = "candidate"
    asyncio.run(trainer.run_initial_training(object()))

    for path in (model_path, calibrator_path, report_path, metadata_path):
        assert path.read_bytes() == good_bundle[path.name]

    failed_status = status.get_training_status()["long_trend"]
    assert failed_status["success"] is False
    assert failed_status["rolled_back"] is True
    assert "evaluation" in (failed_status["error"] or "").lower()

    restarted_after_rollback = LongTrendModel()
    assert restarted_after_rollback.load_model() is True
    assert restarted_after_rollback.is_baseline_mode() is False
    assert restarted_after_rollback.target_type == target_type
    restored_prediction = restarted_after_rollback.predict(good_features)
    assert restarted_after_rollback.last_prediction_was_fallback is False
    assert restored_prediction == pytest.approx(good_prediction, abs=1e-9)
    assert restarted_after_rollback.model.marker == "good"
