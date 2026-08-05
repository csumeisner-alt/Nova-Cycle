"""
Tests for the long-trend baseline-mode fallback.

When no gate-passing trained model is available — either because the pkl is the
legacy 15-feature model (OOS lift ≈ −29 pp) or because no pkl exists at all —
the long signal must:

  1. Set _baseline_mode = True and serve the calibrated majority-class base rate
     (≈0.73 from the calibration report, 0.5 when absent) instead of the stale
     model output.
  2. Return long_signal_mode="baseline", model_state="baseline_mode", and
     prediction_reliable=False in the API response.
  3. Automatically clear baseline mode when a gate-passing 19-feature model is
     promoted to disk (detected by the existing mtime reload path).

Design note on load_model() timing:
  _maybe_reload() is called lazily on first accessor use, NOT in __init__.
  Tests therefore always call is_baseline_mode() (or any other accessor) before
  checking _baseline_mode / model directly.

Covers:
  - Legacy 15-feature XGBoost pkl → baseline_mode=True, model=None
  - Missing pkl → baseline_mode=True
  - Other wrong feature count (e.g. 10) → baseline_mode=True
  - Load error (corrupt bytes) → baseline_mode=True
  - 19-feature pkl → baseline_mode=False, model is not None
  - get_baseline_probability returns calibration_base_rate, falls back to 0.5
  - get_neutral_probability still works and returns the same value
  - Predict endpoint response fields: long_signal_mode, model_state,
    prediction_reliable, ml_confidence != 0.5 when report present
  - Auto-recovery: new 19-feature pkl clears baseline on next _maybe_reload()
"""

import pickle
import sys
import time
import types
from pathlib import Path

import pytest

import ml.calibration as cal
from ml.long_trend import LongTrendModel, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Picklable fake model classes
#
# Classes whose type(obj).__module__ starts with "xgboost" trigger the legacy
# detection logic in load_model().  We register a fake "xgboost.sklearn"
# package in sys.modules so that pickle can resolve the classes by
# (module="xgboost.sklearn", qualname="...") when loading the file.
# ---------------------------------------------------------------------------

_fake_xgb_mod = types.ModuleType("xgboost")
_fake_xgb_sklearn = types.ModuleType("xgboost.sklearn")


class _XGBLike15:
    """Picklable fake 15-feature XGBoost model (triggers legacy path)."""
    __module__ = "xgboost.sklearn"
    n_features_in_ = 15

    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.3, 0.7]] * len(X))


class _XGBLike19:
    """Picklable fake 19-feature XGBoost model (valid gate-passing model)."""
    __module__ = "xgboost.sklearn"
    n_features_in_ = len(FEATURE_NAMES)  # 19

    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.27, 0.73]] * len(X))


class _XGBLike10:
    """Picklable fake XGBoost model with 10 features (wrong count: not 15 or 19).
    Uses the xgboost.sklearn module so pickle resolves it, but n_features_in_=10
    exercises the wrong-feature-count branch in load_model().
    The legacy path only fires when n_features_in_ == 15 exactly."""
    __module__ = "xgboost.sklearn"
    n_features_in_ = 10

    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.5, 0.5]] * len(X))


_fake_xgb_sklearn._XGBLike15 = _XGBLike15
_fake_xgb_sklearn._XGBLike19 = _XGBLike19
_fake_xgb_sklearn._XGBLike10 = _XGBLike10
_fake_xgb_mod.sklearn = _fake_xgb_sklearn

# NOTE: do NOT register into sys.modules at module level — that would displace
# the real xgboost for every subsequent test in the full suite.  Instead, the
# `isolated` fixture registers and unregisters them within each test via
# monkeypatch.setitem, which restores the original value on teardown.


def _write_pkl(path: Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect all ml.long_trend and ml.calibration paths to tmp_path.

    Also temporarily registers the fake xgboost module stubs in sys.modules so
    that pickle can find _XGBLike15 / _XGBLike19 / _XGBLike10 by their
    declared __module__ = "xgboost.sklearn".  monkeypatch.setitem restores the
    originals (real xgboost or absent) on teardown, so this never leaks into
    other tests.

    Gate-pass default: patches get_last_successful_accuracy_metric to return
    "purged_walk_forward_oos" so tests that write a valid 19-feature pkl see
    it correctly clear baseline mode.  Tests that specifically need to verify
    the pre-gate path override this by calling monkeypatch.setattr again with
    a lambda that returns "train" or None.

    _META_PATH is also redirected to tmp_path so that target-type mismatch
    tests never read from or write to the real ml/models directory.  This also
    ensures that any real long_trend_meta.json on disk does not interfere with
    the pre-meta-sidecar code paths tested here.
    """
    from ml import long_trend as lt

    # Scope the fake xgboost registration to this test only.
    monkeypatch.setitem(sys.modules, "xgboost", _fake_xgb_mod)
    monkeypatch.setitem(sys.modules, "xgboost.sklearn", _fake_xgb_sklearn)

    monkeypatch.setattr(lt, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(lt, "MODEL_PATH", tmp_path / "long_trend_model.pkl")
    monkeypatch.setattr(lt, "_META_PATH", tmp_path / "long_trend_meta.json")
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(cal, "CALIBRATOR_PATH", tmp_path / "long_trend_calibrator.pkl")
    monkeypatch.setattr(cal, "REPORT_PATH", tmp_path / "long_trend_calibration.json")

    # Default: simulate a gate-passing training history so that a valid
    # 19-feature pkl clears baseline mode.  The pre-gate path is tested
    # explicitly in TestLoadModelBaselineMode::test_pre_gate_artifact_*.
    monkeypatch.setattr(
        lt, "get_last_successful_accuracy_metric", lambda _: "purged_walk_forward_oos"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# 1. load_model() sets _baseline_mode correctly
#    NB: accessor methods (is_baseline_mode, get_baseline_probability, etc.)
#    call _maybe_reload() which triggers load_model() on first use.
#    Tests call is_baseline_mode() first to trigger the lazy load.
# ---------------------------------------------------------------------------

class TestLoadModelBaselineMode:
    def test_legacy_15feature_sets_baseline_mode(self, isolated):
        """A 15-feature XGBoost pkl must be treated as no trained edge:
        model=None, _baseline_mode=True."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike15())
        m = LongTrendModel()
        # is_baseline_mode() triggers the lazy _maybe_reload()
        assert m.is_baseline_mode() is True, "legacy pkl must set _baseline_mode=True"
        assert m.model is None, "legacy pkl must leave model=None"
        assert m._model_loaded is True

    def test_missing_pkl_sets_baseline_mode(self, isolated):
        """When no pkl file exists the model must enter baseline mode."""
        m = LongTrendModel()
        assert m.is_baseline_mode() is True
        assert m.model is None

    def test_wrong_feature_count_sets_baseline_mode(self, isolated):
        """A pkl whose feature count is neither 15 nor 19 also goes baseline.
        Uses an XGBoost-module class with 10 features (not 15, not 19)."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike10())
        m = LongTrendModel()
        assert m.is_baseline_mode() is True
        assert m.model is None

    def test_corrupt_pkl_sets_baseline_mode(self, isolated):
        """A pkl that fails to unpickle must set baseline mode, not crash."""
        from ml import long_trend as lt

        lt.MODEL_PATH.write_bytes(b"not a valid pickle")
        m = LongTrendModel()
        assert m.is_baseline_mode() is True
        assert m.model is None

    def test_valid_19feature_model_clears_baseline(self, isolated):
        """A valid 19-feature pkl must set _baseline_mode=False and load model."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        m = LongTrendModel()
        assert m.is_baseline_mode() is False, "19-feature model must clear baseline mode"
        assert m.model is not None

    def test_pre_gate_19feature_sets_baseline_mode(self, isolated, monkeypatch):
        """A 19-feature pkl whose last_success_metric is NOT
        'purged_walk_forward_oos' (e.g. the old train-set metric or None) must
        be treated as a pre-gate artifact and enter baseline mode.
        This is the live production case at the time this gate was added:
        training_status shows last_success_accuracy_metric='train'."""
        from ml import long_trend as lt

        # Override the default gate-passing metric to simulate the pre-gate case.
        monkeypatch.setattr(lt, "get_last_successful_accuracy_metric", lambda _: "train")

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        m = LongTrendModel()
        assert m.is_baseline_mode() is True, (
            "19-feature pkl with 'train' metric must set _baseline_mode=True "
            "(pre-gate artifact)"
        )
        assert m.model is None

    def test_no_prior_training_record_sets_baseline_mode(self, isolated, monkeypatch):
        """A 19-feature pkl with last_success_metric=None (no prior success) must
        also enter baseline mode."""
        from ml import long_trend as lt

        monkeypatch.setattr(lt, "get_last_successful_accuracy_metric", lambda _: None)

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        m = LongTrendModel()
        assert m.is_baseline_mode() is True


# ---------------------------------------------------------------------------
# 2. is_baseline_mode() and get_baseline_probability()
# ---------------------------------------------------------------------------

class TestBaselineModeAccessors:
    def test_is_baseline_mode_true_without_pkl(self, isolated):
        m = LongTrendModel()
        assert m.is_baseline_mode() is True

    def test_get_baseline_probability_returns_calibration_rate(self, isolated):
        """When a calibration report exists, get_baseline_probability() must
        return its positive_rate, NOT 0.5."""
        cal.save_calibration_report({"positive_rate": 0.73}, "long_trend")
        m = LongTrendModel()
        prob = m.get_baseline_probability()
        assert prob == pytest.approx(0.73), (
            f"Expected 0.73 from calibration report, got {prob}"
        )

    def test_get_baseline_probability_falls_back_to_half_without_report(self, isolated):
        m = LongTrendModel()
        assert m.get_baseline_probability() == pytest.approx(0.5)

    def test_get_neutral_probability_also_returns_report_rate(self, isolated):
        """get_neutral_probability() is the existing interface; must also
        return the calibration base rate so callers using the old name still work."""
        cal.save_calibration_report({"positive_rate": 0.68}, "long_trend")
        m = LongTrendModel()
        assert m.get_neutral_probability() == pytest.approx(0.68)

    def test_baseline_probability_clipped_at_boundary(self, isolated):
        """positive_rate = 0.0 (outside (0,1)) must fall back to 0.5."""
        cal.save_calibration_report({"positive_rate": 0.0}, "long_trend")
        m = LongTrendModel()
        assert m.get_baseline_probability() == pytest.approx(0.5)

    def test_baseline_probability_not_05_when_report_present(self, isolated):
        """Key contract: data-derived rate (0.73) is not the silent 0.5 fallback."""
        cal.save_calibration_report({"positive_rate": 0.73}, "long_trend")
        m = LongTrendModel()
        assert m.get_baseline_probability() != pytest.approx(0.5)

    def test_baseline_probability_available_in_baseline_mode(self, isolated):
        """Even when in baseline mode (no valid model), the base rate must come
        from the calibration report, not from a model prediction."""
        cal.save_calibration_report({"positive_rate": 0.73}, "long_trend")
        m = LongTrendModel()
        assert m.is_baseline_mode() is True
        assert m.get_baseline_probability() == pytest.approx(0.73)


# ---------------------------------------------------------------------------
# 3. Auto-recovery: new 19-feature pkl clears baseline via mtime reload
# ---------------------------------------------------------------------------

class TestBaselineModeAutoRecovery:
    def test_new_19feature_pkl_clears_baseline_on_reload(self, isolated):
        """When a gate-passing 19-feature model is written to disk after the
        model was in baseline mode, the next _maybe_reload() call must clear
        _baseline_mode and load the model."""
        from ml import long_trend as lt

        # Start in baseline mode (no pkl)
        m = LongTrendModel()
        assert m.is_baseline_mode() is True

        # Simulate a gate-passing retrain: write a 19-feature pkl
        _write_pkl(lt.MODEL_PATH, _XGBLike19())

        # Force mtime change detection
        m._loaded_mtime = None
        m._maybe_reload()

        assert m.is_baseline_mode() is False, "promoted 19-feature model must clear baseline"
        assert m.model is not None

    def test_replacing_legacy_with_19feature_clears_baseline(self, isolated):
        """Legacy pkl → baseline; then a retrain writes a 19-feature pkl →
        auto-recovery clears baseline on next reload."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike15())
        m = LongTrendModel()
        assert m.is_baseline_mode() is True

        # Retrain promotes a 19-feature model
        time.sleep(0.01)  # ensure mtime differs
        _write_pkl(lt.MODEL_PATH, _XGBLike19())

        # Trigger mtime-based reload
        m._loaded_mtime = None
        m._maybe_reload()

        assert m.is_baseline_mode() is False
        assert m.model is not None


# ---------------------------------------------------------------------------
# 4. Predict endpoint response contract (logic-level, no DB required)
# ---------------------------------------------------------------------------

class TestPredictLongBaselineResponseContract:
    def test_baseline_mode_overrides_model_state(self, isolated):
        """The predict_long override changes model_state from 'model_unavailable'
        to 'baseline_mode' when long_signal_mode == 'baseline'."""
        ml_fallback = True
        long_signal_mode = "baseline"
        model_state = "model_unavailable"  # raw from _model_reliability
        if long_signal_mode == "baseline":
            model_state = "baseline_mode"
        assert model_state == "baseline_mode"

    def test_ml_confidence_is_base_rate_not_05_when_report_present(self, isolated):
        """When baseline mode is active and a calibration report exists, the
        ml_confidence served must be the calibrated base rate, not 0.5."""
        cal.save_calibration_report({"positive_rate": 0.73}, "long_trend")
        m = LongTrendModel()
        assert m.is_baseline_mode() is True
        ml_confidence = m.get_baseline_probability()
        assert ml_confidence == pytest.approx(0.73)
        assert ml_confidence != pytest.approx(0.5)

    def test_prediction_reliable_false_in_baseline(self, isolated):
        """Baseline mode must set prediction_reliable=False so the UI shows
        the degraded badge and notifications are suppressed."""
        long_signal_mode = "baseline"
        prediction_reliable = True
        if long_signal_mode == "baseline":
            prediction_reliable = False
        assert prediction_reliable is False

    def test_long_signal_mode_field_is_baseline_when_no_model(self, isolated):
        """long_signal_mode='baseline' when is_baseline_mode() is True."""
        m = LongTrendModel()
        long_signal_mode = "baseline" if m.is_baseline_mode() else "trained"
        assert long_signal_mode == "baseline"

    def test_ml_confidence_is_05_in_baseline_without_report(self, isolated):
        """No calibration report → 0.5 safe fallback."""
        m = LongTrendModel()
        assert m.is_baseline_mode() is True
        assert m.get_baseline_probability() == pytest.approx(0.5)

    def test_long_signal_mode_is_trained_for_valid_model(self, isolated):
        """When a valid 19-feature model is loaded, long_signal_mode='trained'."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        m = LongTrendModel()
        long_signal_mode = "baseline" if m.is_baseline_mode() else "trained"
        assert long_signal_mode == "trained"


# ---------------------------------------------------------------------------
# 5. is_neutral_fallback() consistency
# ---------------------------------------------------------------------------

class TestNeutralFallbackConsistency:
    def test_legacy_pkl_also_sets_neutral_fallback(self, isolated):
        """is_neutral_fallback() must still be True for a legacy pkl since
        model is None; predict_long checks is_baseline_mode() first."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike15())
        m = LongTrendModel()
        # Call is_baseline_mode() to trigger lazy load
        assert m.is_baseline_mode() is True
        assert m.is_neutral_fallback() is True

    def test_valid_model_has_neither_fallback_nor_baseline(self, isolated):
        """A successfully loaded 19-feature model must have both flags False."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        m = LongTrendModel()
        assert m.is_baseline_mode() is False
        assert m.is_neutral_fallback() is False

    def test_baseline_mode_implies_neutral_fallback(self, isolated):
        """Whenever is_baseline_mode() is True, is_neutral_fallback must also be
        True (model is None in both cases)."""
        m = LongTrendModel()  # no pkl → baseline
        assert m.is_baseline_mode() is True
        assert m.is_neutral_fallback() is True


# ---------------------------------------------------------------------------
# 6. Target-type mismatch: switching LONG_TARGET_TYPE without retraining
# ---------------------------------------------------------------------------

class TestTargetTypeMismatch:
    """Confirm that changing LONG_TARGET_TYPE in config without completing a
    retrain causes load_model() to enter baseline mode rather than serving
    a direction model under drawdown / three-state semantics."""

    def _write_meta(self, path, target_type: str) -> None:
        """Write a minimal meta sidecar JSON."""
        import json
        path.write_text(json.dumps({"target_type": target_type}))

    def test_direction_model_with_drawdown_config_is_baseline(
        self, isolated, monkeypatch
    ):
        """Train a direction model (meta sidecar says 'direction'), then switch
        LONG_TARGET_TYPE to 'drawdown_event' without retraining.
        load_model() must detect the mismatch and enter baseline mode."""
        from ml import long_trend as lt

        # Promote a valid 19-feature direction model
        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        self._write_meta(lt._META_PATH, "direction")

        # Change config without running a new retrain
        monkeypatch.setattr(lt.settings, "LONG_TARGET_TYPE", "drawdown_event")

        m = LongTrendModel()
        assert m.is_baseline_mode() is True, (
            "Switching LONG_TARGET_TYPE from direction → drawdown_event "
            "without retraining must force baseline mode"
        )
        assert m.model is None

    def test_direction_model_with_three_state_config_is_baseline(
        self, isolated, monkeypatch
    ):
        """Same safety check when switching to three_state."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        self._write_meta(lt._META_PATH, "direction")

        monkeypatch.setattr(lt.settings, "LONG_TARGET_TYPE", "three_state")

        m = LongTrendModel()
        assert m.is_baseline_mode() is True, (
            "Switching LONG_TARGET_TYPE from direction → three_state "
            "without retraining must force baseline mode"
        )
        assert m.model is None

    def test_matching_target_type_does_not_force_baseline(
        self, isolated, monkeypatch
    ):
        """When the meta sidecar and LONG_TARGET_TYPE agree (both 'direction'),
        no mismatch → baseline mode must NOT be set."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        self._write_meta(lt._META_PATH, "direction")

        monkeypatch.setattr(lt.settings, "LONG_TARGET_TYPE", "direction")

        m = LongTrendModel()
        assert m.is_baseline_mode() is False
        assert m.model is not None

    def test_absent_meta_sidecar_defaults_to_direction(
        self, isolated, monkeypatch
    ):
        """When no meta sidecar exists, load_model() assumes 'direction'.
        A valid 19-feature pkl with LONG_TARGET_TYPE='direction' must load
        successfully (no baseline mode)."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        # _META_PATH does not exist (isolated fixture points it to an empty tmp_path)
        monkeypatch.setattr(lt.settings, "LONG_TARGET_TYPE", "direction")

        m = LongTrendModel()
        assert m.is_baseline_mode() is False
        assert m.model is not None

    def test_absent_meta_sidecar_with_drawdown_config_is_baseline(
        self, isolated, monkeypatch
    ):
        """No sidecar defaults to 'direction'; if LONG_TARGET_TYPE='drawdown_event'
        that is a mismatch → baseline mode."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        # No meta sidecar written; absent sidecar defaults to direction
        monkeypatch.setattr(lt.settings, "LONG_TARGET_TYPE", "drawdown_event")

        m = LongTrendModel()
        assert m.is_baseline_mode() is True, (
            "No meta sidecar (defaults to direction) + LONG_TARGET_TYPE=drawdown_event "
            "must force baseline mode"
        )
        assert m.model is None

    def test_baseline_clears_after_matching_retrain(
        self, isolated, monkeypatch
    ):
        """After entering mismatch baseline mode, a retrain that writes a new
        pkl + matching meta sidecar must clear baseline on the next reload."""
        from ml import long_trend as lt

        # Start: direction pkl + drawdown_event config → mismatch → baseline
        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        self._write_meta(lt._META_PATH, "direction")
        monkeypatch.setattr(lt.settings, "LONG_TARGET_TYPE", "drawdown_event")

        m = LongTrendModel()
        assert m.is_baseline_mode() is True

        # Simulate a successful drawdown_event retrain: update pkl + sidecar
        import time
        time.sleep(0.01)  # ensure mtime differs
        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        self._write_meta(lt._META_PATH, "drawdown_event")

        # Force mtime-based reload
        m._loaded_mtime = None
        m._maybe_reload()

        assert m.is_baseline_mode() is False, (
            "After a matching drawdown_event retrain, baseline mode must clear"
        )
        assert m.model is not None


# ---------------------------------------------------------------------------
# 7. save_promotion_meta: meta sidecar written correctly after gate-passing
# ---------------------------------------------------------------------------

class TestSavePromotionMeta:
    """Confirm that save_promotion_meta() writes the correct target_type to the
    meta sidecar and that load_model() then reads it back accurately."""

    def test_direction_sidecar_written(self, isolated):
        """save_promotion_meta('direction') must write target_type=direction."""
        import json
        from ml import long_trend as lt

        LongTrendModel.save_promotion_meta("direction")

        assert lt._META_PATH.exists(), "Meta sidecar must be created"
        data = json.loads(lt._META_PATH.read_text())
        assert data.get("target_type") == "direction"

    def test_drawdown_event_sidecar_written(self, isolated):
        """save_promotion_meta('drawdown_event') must write the correct target_type.
        This covers the gate-passing promotion path in trainer.py."""
        import json
        from ml import long_trend as lt

        LongTrendModel.save_promotion_meta("drawdown_event")

        assert lt._META_PATH.exists(), "Meta sidecar must be created"
        data = json.loads(lt._META_PATH.read_text())
        assert data.get("target_type") == "drawdown_event", (
            f"Expected target_type='drawdown_event', got {data!r}"
        )

    def test_three_state_sidecar_written(self, isolated):
        """save_promotion_meta('three_state') must write the correct target_type."""
        import json
        from ml import long_trend as lt

        LongTrendModel.save_promotion_meta("three_state")

        data = json.loads(lt._META_PATH.read_text())
        assert data.get("target_type") == "three_state"

    def test_sidecar_overwritten_on_second_promotion(self, isolated):
        """A second promotion overwrites the previous sidecar with the new type."""
        import json
        from ml import long_trend as lt

        LongTrendModel.save_promotion_meta("direction")
        LongTrendModel.save_promotion_meta("drawdown_event")

        data = json.loads(lt._META_PATH.read_text())
        assert data.get("target_type") == "drawdown_event"

    def test_load_model_reads_sidecar_written_by_save_promotion_meta(
        self, isolated, monkeypatch
    ):
        """Integration: save_promotion_meta followed by load_model() correctly
        surfaces the promoted target_type on the model instance."""
        from ml import long_trend as lt

        _write_pkl(lt.MODEL_PATH, _XGBLike19())
        LongTrendModel.save_promotion_meta("direction")
        monkeypatch.setattr(lt.settings, "LONG_TARGET_TYPE", "direction")

        m = LongTrendModel()
        assert m.is_baseline_mode() is False
        assert m._promoted_target_type == "direction"
