"""
Integration-level tests for ModelTrainer._track_and_alert_baseline_duration.

Unlike the unit tests in test_baseline_duration_alert.py (which exercise
training_status.py in isolation), these tests call the trainer method directly
and verify:

  1. FCMNotifier.send_baseline_duration_alert is called exactly once when the
     model has been in baseline mode past the alert threshold.
  2. A second call with the same state does NOT fire the alert again
     (mark_baseline_duration_alert_sent prevents a duplicate).
  3. After clear_baseline_mode_tracking (simulating a gate-passing retrain)
     followed by a new aged onset, the alert fires once more (re-armed).
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ml.training_status as ts
from ml.trainer import ModelTrainer
from ml.training_status import (
    clear_baseline_mode_tracking,
    record_baseline_mode_onset,
    record_training_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aged_onset(days: float) -> str:
    """Return an ISO timestamp ``days`` days in the past."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _set_onset(status_path, model_name: str, onset_iso: str) -> None:
    """Overwrite the baseline_mode_since field in the status file."""
    data = json.loads(status_path.read_text())
    data[model_name]["baseline_mode_since"] = onset_iso
    status_path.write_text(json.dumps(data))


def _make_db_session(fake_token: str = "fake-device-token"):
    """Return a mock AsyncSession that yields one DeviceToken row."""
    device = MagicMock()
    device.token = fake_token

    scalars_mock = MagicMock()
    scalars_mock.all.return_value = [device]

    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock

    db_session = AsyncMock()
    db_session.execute.return_value = result_mock
    return db_session


def _make_trainer(in_baseline: bool) -> ModelTrainer:
    """Build a ModelTrainer without running __init__, mocking long_model only."""
    trainer = ModelTrainer.__new__(ModelTrainer)
    long_model_mock = MagicMock()
    long_model_mock.is_baseline_mode.return_value = in_baseline
    trainer.long_model = long_model_mock
    return trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    """Redirect training_status to an isolated tmp file and seed a long_trend entry."""
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")
    record_training_result("long_trend", success=False, error="gate_fail")


@pytest.fixture
def low_threshold(monkeypatch):
    """Set LONG_BASELINE_MODE_ALERT_DAYS to 7 so a 10-day onset trips the gate."""
    from config import settings
    monkeypatch.setattr(settings, "LONG_BASELINE_MODE_ALERT_DAYS", 7)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_alert_fires_once_on_first_eligible_call(
    isolated_status, low_threshold
):
    """First call with aged onset fires send_baseline_duration_alert exactly once."""
    # Onset 10 days ago — past the 7-day threshold.
    _set_onset(ts.STATUS_PATH, "long_trend", _aged_onset(10))

    trainer = _make_trainer(in_baseline=True)
    db_session = _make_db_session()

    mock_notifier = AsyncMock()
    mock_notifier.send_baseline_duration_alert.return_value = True

    with patch("notifications.fcm.FCMNotifier", return_value=mock_notifier):
        await trainer._track_and_alert_baseline_duration(db_session, "long_trend")

    mock_notifier.send_baseline_duration_alert.assert_called_once()


async def test_alert_not_called_again_after_first_fire(
    isolated_status, low_threshold
):
    """Second call in the same baseline episode does NOT fire the alert again."""
    _set_onset(ts.STATUS_PATH, "long_trend", _aged_onset(10))

    trainer = _make_trainer(in_baseline=True)
    db_session = _make_db_session()

    mock_notifier = AsyncMock()
    mock_notifier.send_baseline_duration_alert.return_value = True

    with patch("notifications.fcm.FCMNotifier", return_value=mock_notifier):
        # First call — fires.
        await trainer._track_and_alert_baseline_duration(db_session, "long_trend")
        assert mock_notifier.send_baseline_duration_alert.call_count == 1

        # Second call — must not fire (already sent flag is set).
        mock_notifier.send_baseline_duration_alert.reset_mock()
        await trainer._track_and_alert_baseline_duration(db_session, "long_trend")
        mock_notifier.send_baseline_duration_alert.assert_not_called()


async def test_alert_rearmed_after_gate_passing_retrain(
    isolated_status, low_threshold
):
    """After clear_baseline_mode_tracking + new aged onset, alert fires again."""
    # First episode: alert already sent.
    _set_onset(ts.STATUS_PATH, "long_trend", _aged_onset(10))
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_alert_sent"] = True
    ts.STATUS_PATH.write_text(json.dumps(data))

    trainer = _make_trainer(in_baseline=True)
    db_session = _make_db_session()

    mock_notifier = AsyncMock()
    mock_notifier.send_baseline_duration_alert.return_value = True

    with patch("notifications.fcm.FCMNotifier", return_value=mock_notifier):
        # Gate-passing retrain clears the episode.
        clear_baseline_mode_tracking("long_trend")

        # New baseline episode begins.
        record_baseline_mode_onset("long_trend")
        # Age the new onset past the threshold.
        _set_onset(ts.STATUS_PATH, "long_trend", _aged_onset(10))

        # Third call — re-armed, should fire exactly once.
        await trainer._track_and_alert_baseline_duration(db_session, "long_trend")
        mock_notifier.send_baseline_duration_alert.assert_called_once()
