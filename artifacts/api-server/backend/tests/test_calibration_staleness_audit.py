"""
Integration tests for audit_calibration_report_staleness (ml/trainer.py).

The audit function is called during retrain_if_needed whenever retraining is
skipped.  It compares the DB's current labeled-row count against the number
recorded in the on-disk calibration report and calls
mark_calibration_report_stale when the report significantly under-counts what
is now in the DB.

Tests cover:
  - report is marked stale with real dataset metadata when DB has 2×+ rows
  - idempotence: audit is a no-op when the report is already stale with
    current DB counts
  - a subsequent fresh save (from a real retrain) clears stale=false
  - a fresh evaluated=true report is never marked stale by the audit
  - the audit is silent (returns False) when DB has fewer rows than threshold
  - the audit never raises even on a broken db_session
  - retrain_if_needed calls the audit when skipping retraining
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
import pytest

from ml import calibration as cal
from ml.trainer import (
    audit_calibration_report_staleness,
    _extract_report_labeled_rows,
    _fetch_daily_candle_meta,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cal(tmp_path, monkeypatch):
    """Redirect all calibration writes to a tmp directory."""
    monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(cal, "CALIBRATOR_PATH", tmp_path / "long_trend_calibrator.pkl")
    monkeypatch.setattr(cal, "REPORT_PATH", tmp_path / "long_trend_calibration.json")
    return tmp_path


def _mock_meta(labeled_rows: int = 1633, total_candles: int = 2521):
    """Return an async function that simulates _fetch_daily_candle_meta."""
    async def _fetch(db_session):
        return {
            "total_candles": total_candles,
            "labeled_rows": labeled_rows,
            "date_start": "2016-07-25",
            "date_end": "2026-08-04",
            "note": "VOO daily regular-hours candles; labeled = rows where |21-day return| >= 2%",
        }
    return _fetch


def _mock_meta_none():
    async def _fetch(db_session):
        return None
    return _fetch


# ─────────────────────────────────────────────────────────────────────────────
# _extract_report_labeled_rows helper
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_labeled_rows_from_dataset_key():
    report = {"dataset": {"labeled_rows": 101}}
    assert _extract_report_labeled_rows(report) == 101


def test_extract_labeled_rows_from_reason_string():
    report = {"reason": "not enough rows (101) for walk-forward"}
    assert _extract_report_labeled_rows(report) == 101


def test_extract_labeled_rows_prefers_dataset_over_reason():
    report = {
        "dataset": {"labeled_rows": 200},
        "reason": "not enough rows (101) for walk-forward",
    }
    assert _extract_report_labeled_rows(report) == 200


def test_extract_labeled_rows_returns_zero_when_absent():
    assert _extract_report_labeled_rows({}) == 0
    assert _extract_report_labeled_rows({"dataset": {}}) == 0


# ─────────────────────────────────────────────────────────────────────────────
# audit_calibration_report_staleness — core transitions
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_marks_stale_when_db_has_2x_rows(tmp_cal, monkeypatch):
    """A report produced with 101 rows should be marked stale when the DB
    now holds 1633 labeled rows (16× more)."""
    import ml.trainer as trainer_mod

    cal.save_calibration_report({
        "evaluated": False,
        "reason": "not enough rows (101) for walk-forward",
        "dataset": {"labeled_rows": 101, "total_candles": 200,
                    "date_start": "2025-06-02", "date_end": "2026-03-06"},
    })
    assert cal.get_calibration_report()["stale"] is False

    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta())

    result = asyncio.run(audit_calibration_report_staleness(None))

    assert result is True
    report = cal.get_calibration_report()
    assert report["stale"] is True
    assert report["dataset"]["labeled_rows"] == 1633
    assert report["dataset"]["total_candles"] == 2521
    assert report["dataset"]["date_start"] == "2016-07-25"
    assert report["dataset"]["date_end"] == "2026-08-04"
    assert "1633" in report["stale_note"]
    assert "2521" in report["stale_note"]
    assert "marked_stale_at" in report


def test_audit_is_idempotent_when_already_stale_with_current_counts(tmp_cal, monkeypatch):
    """When the report is already stale with the exact current DB counts, the
    audit must not rewrite the file (idempotence check via mtime)."""
    import ml.trainer as trainer_mod
    import os

    cal.save_calibration_report({"evaluated": False, "reason": "not enough rows (101)"})
    cal.mark_calibration_report_stale(
        dataset_meta={"labeled_rows": 1633, "total_candles": 2521,
                      "date_start": "2016-07-25", "date_end": "2026-08-04"},
    )

    path = cal.calibration_report_path("long_trend")
    mtime_before = os.stat(path).st_mtime

    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta())
    result = asyncio.run(audit_calibration_report_staleness(None))

    assert result is False  # idempotent: already stale with these counts
    assert os.stat(path).st_mtime == mtime_before  # file not rewritten


def test_audit_updates_stale_counts_when_db_has_grown_further(tmp_cal, monkeypatch):
    """If a report was marked stale with 800 labeled rows but the DB now has
    1633, the audit should update the stale record with fresh counts."""
    import ml.trainer as trainer_mod

    cal.save_calibration_report({"evaluated": False, "reason": "not enough rows (101)"})
    cal.mark_calibration_report_stale(
        dataset_meta={"labeled_rows": 800, "total_candles": 1200},
    )

    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta())
    result = asyncio.run(audit_calibration_report_staleness(None))

    assert result is True
    report = cal.get_calibration_report()
    assert report["dataset"]["labeled_rows"] == 1633
    assert report["dataset"]["total_candles"] == 2521


def test_fresh_save_after_audit_clears_stale(tmp_cal, monkeypatch):
    """A fresh save_calibration_report (simulating a completed retrain) must
    clear stale=false regardless of what the audit previously set."""
    import ml.trainer as trainer_mod

    cal.save_calibration_report({"evaluated": False, "reason": "not enough rows (101)",
                                  "dataset": {"labeled_rows": 101, "total_candles": 200}})
    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta())
    asyncio.run(audit_calibration_report_staleness(None))
    assert cal.get_calibration_report()["stale"] is True

    # Simulate a successful retrain writing a fresh report
    cal.save_calibration_report({"evaluated": True, "oos_accuracy": 0.57,
                                  "dataset": {"labeled_rows": 1633, "total_candles": 2521}})
    report = cal.get_calibration_report()
    assert report["stale"] is False
    assert "stale_note" not in report
    assert "marked_stale_at" not in report
    assert cal.is_calibration_report_stale() is False


def test_fresh_evaluated_report_is_not_marked_stale(tmp_cal, monkeypatch):
    """A report with evaluated=True is intentional and must never be marked
    stale by the audit, even if the DB has many more rows."""
    import ml.trainer as trainer_mod

    cal.save_calibration_report({
        "evaluated": True,
        "oos_accuracy": 0.55,
        "dataset": {"labeled_rows": 300, "total_candles": 400},
    })
    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta())

    result = asyncio.run(audit_calibration_report_staleness(None))

    assert result is False
    assert cal.get_calibration_report()["stale"] is False


def test_audit_skips_when_db_too_small(tmp_cal, monkeypatch):
    """When the DB has fewer labeled rows than min_db_rows_to_flag (500),
    the audit must not mark the report stale — the DB itself may be in
    a near-empty initial-deployment state."""
    import ml.trainer as trainer_mod

    cal.save_calibration_report({"evaluated": False, "reason": "not enough rows (50)"})
    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta(labeled_rows=300))

    result = asyncio.run(audit_calibration_report_staleness(None))
    assert result is False


def test_audit_skips_when_db_not_significantly_larger(tmp_cal, monkeypatch):
    """DB only 1.5× the report count (below the 2× threshold) → no stale mark."""
    import ml.trainer as trainer_mod

    cal.save_calibration_report({
        "evaluated": False,
        "dataset": {"labeled_rows": 800, "total_candles": 1000},
    })
    # 1100 labeled rows = 1.375× 800 → below the 2.0 threshold
    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta(labeled_rows=1100, total_candles=1400))

    result = asyncio.run(audit_calibration_report_staleness(None))
    assert result is False


def test_audit_returns_false_when_no_report_exists(tmp_cal, monkeypatch):
    """No report on disk → audit is a no-op (returns False, no file created)."""
    import ml.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta())
    result = asyncio.run(audit_calibration_report_staleness(None))
    assert result is False
    assert not cal.calibration_report_path("long_trend").exists()


def test_audit_never_raises_on_broken_db_session(tmp_cal, monkeypatch):
    """A failing _fetch_daily_candle_meta must not propagate an exception —
    the audit function swallows all errors."""
    import ml.trainer as trainer_mod

    async def _broken(_db):
        raise RuntimeError("DB connection refused")

    cal.save_calibration_report({"evaluated": False, "reason": "not enough rows (101)"})
    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _broken)

    result = asyncio.run(audit_calibration_report_staleness(None))
    assert result is False  # swallowed silently


# ─────────────────────────────────────────────────────────────────────────────
# retrain_if_needed calls the audit when skipping
# ─────────────────────────────────────────────────────────────────────────────

def test_retrain_if_needed_calls_audit_when_not_retraining(tmp_cal, monkeypatch):
    """When retrain_if_needed decides no retraining is needed it must invoke
    audit_calibration_report_staleness so stale reports are repaired without
    waiting for a full retrain cycle."""
    from ml import long_trend as lt
    from ml import short_trend as st
    from ml import training_status as ts
    from ml.trainer import ModelTrainer

    tmp_path = tmp_cal  # same monkeypatched path

    # Redirect model paths away from real files
    monkeypatch.setattr(lt, "MODEL_PATH", tmp_path / "long_trend_model.pkl")
    monkeypatch.setattr(lt, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(st, "MODEL_PATH", tmp_path / "short_trend_model.pkl")
    monkeypatch.setattr(st, "MODEL_DIR", tmp_path)

    # Redirect training-status writes so record_training_result() never
    # touches the real ml/models/training_status.json.
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")

    import ml.trainer as trainer_mod

    # Seed a stale-candidate report (evaluated=False, too few rows)
    cal.save_calibration_report({
        "evaluated": False,
        "reason": "not enough rows (101) for walk-forward",
        "dataset": {"labeled_rows": 101, "total_candles": 200},
    })

    # Models were last trained "recently" so retrain_if_needed will skip
    from datetime import datetime, timezone
    ts.record_training_result("long_trend", success=True, accuracy=0.65)
    ts.record_training_result("short_trend", success=True, accuracy=0.65)

    monkeypatch.setattr(trainer_mod, "_fetch_daily_candle_meta", _mock_meta())

    # Stub out _get_last_trained to return "just now" so retrain is skipped
    async def _recent(_db):
        return datetime.utcnow()

    monkeypatch.setattr(ModelTrainer, "_get_last_trained", staticmethod(_recent))

    # Stub _missing_model_files so it thinks models exist
    monkeypatch.setattr(ModelTrainer, "_missing_model_files", lambda self: [])

    trainer = ModelTrainer()
    retrained = asyncio.run(trainer.retrain_if_needed(object()))

    assert retrained is False  # confirmed: no retrain happened
    report = cal.get_calibration_report()
    assert report["stale"] is True, (
        "retrain_if_needed must have called audit, which must have marked the report stale"
    )
    assert report["dataset"]["labeled_rows"] == 1633
