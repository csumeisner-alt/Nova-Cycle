"""Training-stuck push alert tests: the alert must arm once the
consecutive-failure count crosses the threshold, fire only once per stuck
episode, and re-arm after a successful retrain."""

import pytest

from ml import training_status as ts
from ml.training_status import (
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD,
    mark_stuck_alert_sent,
    record_training_result,
    should_send_stuck_alert,
)


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")


def _fail_n(model, n):
    for _ in range(n):
        record_training_result(model, success=False, error="boom")


def test_not_armed_below_threshold(isolated_status):
    _fail_n("long_trend", CONSECUTIVE_FAILURE_ALERT_THRESHOLD - 1)
    assert should_send_stuck_alert("long_trend") is False


def test_armed_at_threshold(isolated_status):
    _fail_n("long_trend", CONSECUTIVE_FAILURE_ALERT_THRESHOLD)
    assert should_send_stuck_alert("long_trend") is True


def test_fires_once_per_episode(isolated_status):
    _fail_n("long_trend", CONSECUTIVE_FAILURE_ALERT_THRESHOLD)
    mark_stuck_alert_sent("long_trend")
    assert should_send_stuck_alert("long_trend") is False
    # More failures in the same episode do not re-fire.
    _fail_n("long_trend", 2)
    assert should_send_stuck_alert("long_trend") is False


def test_success_rearms_alert(isolated_status):
    _fail_n("long_trend", CONSECUTIVE_FAILURE_ALERT_THRESHOLD)
    mark_stuck_alert_sent("long_trend")
    record_training_result("long_trend", success=True, accuracy=0.7)
    assert should_send_stuck_alert("long_trend") is False  # counter reset
    # A new episode fires again once it crosses the threshold.
    _fail_n("long_trend", CONSECUTIVE_FAILURE_ALERT_THRESHOLD)
    assert should_send_stuck_alert("long_trend") is True


def test_unsent_alert_stays_armed(isolated_status):
    # If sending fails (no mark), further failed attempts keep it armed.
    _fail_n("short_trend", CONSECUTIVE_FAILURE_ALERT_THRESHOLD + 1)
    assert should_send_stuck_alert("short_trend") is True


def test_no_entry_not_armed(isolated_status):
    assert should_send_stuck_alert("long_trend") is False


def test_mark_without_entry_is_noop(isolated_status):
    mark_stuck_alert_sent("long_trend")  # must not raise
    assert should_send_stuck_alert("long_trend") is False
