"""
Tests for the short-trend model's rollback sidecar round-trip.

The short-trend model has three sidecars:
  1. walk-forward report  (_walkforward_report_path("short_trend"))
  2. calibrator           (calibrator_path("short_trend"))
  3. calibration report   (calibration_report_path("short_trend"))

These tests verify that _backup_model_file + _restore_model_file correctly
round-trips all three sidecars when a short-trend retrain is flagged and rolled
back, mirroring the equivalent coverage for the long-trend meta sidecar.
"""

import pickle
from pathlib import Path

import pytest

import ml.calibration as cal
import ml.short_trend as st
from ml.trainer import _backup_model_file, _restore_model_file


# ---------------------------------------------------------------------------
# Fixture: redirect all short_trend and calibration paths to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_short(tmp_path, monkeypatch):
    """Redirect short_trend MODEL_PATH/MODEL_DIR and ml.calibration.MODEL_DIR
    to tmp_path so no test touches the real ml/models directory.

    _sidecar_files() calls calibrator_path("short_trend"),
    _walkforward_report_path("short_trend"), and
    calibration_report_path("short_trend"), all of which read MODEL_DIR at
    call-time — patching MODEL_DIR here is sufficient.
    """
    monkeypatch.setattr(st, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(st, "MODEL_PATH", tmp_path / "short_trend_model.pkl")
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_model_path(tmp_path: Path) -> Path:
    return tmp_path / "short_trend_model.pkl"


def _sidecar_paths(tmp_path: Path):
    """Return the three sidecar paths as they will be resolved by _sidecar_files."""
    return (
        tmp_path / "short_trend_walkforward.json",   # _walkforward_report_path
        tmp_path / "short_trend_calibrator.pkl",      # calibrator_path
        tmp_path / "short_trend_calibration.json",    # calibration_report_path
    )


# ---------------------------------------------------------------------------
# Core round-trip test
# ---------------------------------------------------------------------------

class TestShortTrendSidecarRollback:
    """_backup_model_file + _restore_model_file must restore all three
    short-trend sidecars to their pre-retrain content."""

    def test_all_three_sidecars_are_restored(self, isolated_short):
        """Full round-trip:
          1. Write good short-trend pkl + all three sidecars.
          2. Backup via _backup_model_file.
          3. Overwrite pkl + sidecars with regressed-retrain content.
          4. Restore via _restore_model_file.
          5. Assert all three sidecars contain the original good content.
        """
        tmp_path = isolated_short
        model_path = _short_model_path(tmp_path)
        walkforward, calibrator, cal_report = _sidecar_paths(tmp_path)

        # --- Step 1: write the "good" pre-retrain files ---
        model_path.write_bytes(b"good-short-model")
        walkforward.write_text('{"good": "walkforward"}')
        calibrator.write_bytes(pickle.dumps({"good": "calibrator"}))
        cal_report.write_text('{"positive_rate": 0.62}')

        # --- Step 2: backup ---
        backup_path = _backup_model_file(model_path)
        assert backup_path is not None, "_backup_model_file must return a backup path"
        assert backup_path.exists(), "Backup pkl must exist after _backup_model_file"

        # Confirm sidecar backups exist
        assert walkforward.with_suffix(walkforward.suffix + ".bak").exists(), (
            "walk-forward report backup must be created"
        )
        assert calibrator.with_suffix(calibrator.suffix + ".bak").exists(), (
            "calibrator backup must be created"
        )
        assert cal_report.with_suffix(cal_report.suffix + ".bak").exists(), (
            "calibration report backup must be created"
        )

        # --- Step 3: overwrite with regressed-retrain content ---
        model_path.write_bytes(b"regressed-short-model")
        walkforward.write_text('{"regressed": "walkforward"}')
        calibrator.write_bytes(pickle.dumps({"regressed": "calibrator"}))
        cal_report.write_text('{"positive_rate": 0.99}')

        # --- Step 4: restore ---
        restored = _restore_model_file(model_path, backup_path, "short_trend")
        assert restored is True, "_restore_model_file must return True on success"

        # --- Step 5: verify the pkl and all three sidecars are back ---
        assert model_path.read_bytes() == b"good-short-model", (
            "pkl must be restored to its pre-retrain bytes"
        )
        assert walkforward.read_text() == '{"good": "walkforward"}', (
            "walk-forward report must be restored to its pre-retrain content"
        )
        assert pickle.loads(calibrator.read_bytes()) == {"good": "calibrator"}, (
            "calibrator must be restored to its pre-retrain content"
        )
        assert cal_report.read_text() == '{"positive_rate": 0.62}', (
            "calibration report must be restored to its pre-retrain content"
        )

    def test_absent_sidecar_is_removed_after_restore(self, isolated_short):
        """When a sidecar did not exist at backup time but the regressed retrain
        wrote one, _restore_model_file must delete the regressed sidecar so the
        restored state exactly matches the pre-retrain state.
        """
        tmp_path = isolated_short
        model_path = _short_model_path(tmp_path)
        walkforward, calibrator, cal_report = _sidecar_paths(tmp_path)

        # Pre-retrain: model + only the calibration report; no walk-forward or
        # calibrator yet.
        model_path.write_bytes(b"good-short-model-no-sidecars")
        cal_report.write_text('{"positive_rate": 0.55}')
        # walkforward and calibrator do NOT exist yet

        backup_path = _backup_model_file(model_path)
        assert backup_path is not None

        # Regressed retrain writes all three sidecars
        model_path.write_bytes(b"regressed-short-model")
        walkforward.write_text('{"regressed": "walkforward"}')
        calibrator.write_bytes(b"regressed-calibrator")
        cal_report.write_text('{"positive_rate": 0.99}')

        restored = _restore_model_file(model_path, backup_path, "short_trend")
        assert restored is True

        # pkl restored
        assert model_path.read_bytes() == b"good-short-model-no-sidecars"
        # walk-forward was absent before backup → must be deleted after restore
        assert not walkforward.exists(), (
            "walk-forward report written by regressed retrain must be removed on rollback"
        )
        # calibrator was absent before backup → must be deleted after restore
        assert not calibrator.exists(), (
            "calibrator written by regressed retrain must be removed on rollback"
        )
        # calibration report was present → must be restored
        assert cal_report.read_text() == '{"positive_rate": 0.55}', (
            "calibration report must be restored to its pre-retrain content"
        )

    def test_no_backup_returns_false(self, isolated_short):
        """When no backup exists (first-ever retrain), _restore_model_file must
        return False without crashing."""
        tmp_path = isolated_short
        model_path = _short_model_path(tmp_path)
        model_path.write_bytes(b"some-model")

        result = _restore_model_file(model_path, None, "short_trend")
        assert result is False, (
            "_restore_model_file must return False when backup_path is None"
        )

    def test_backup_returns_none_when_no_pkl(self, isolated_short):
        """When the short-trend pkl does not exist, _backup_model_file must
        return None (no backup possible)."""
        tmp_path = isolated_short
        model_path = _short_model_path(tmp_path)
        # model_path intentionally not written

        result = _backup_model_file(model_path)
        assert result is None, (
            "_backup_model_file must return None when the model file is absent"
        )
