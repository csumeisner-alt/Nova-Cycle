"""Consecutive-failure alert tests: training_status must count consecutive
failed retrain attempts per model, expose them for the health endpoint, and
reset the counter on a successful retrain."""

import pytest

from ml import training_status as ts
from ml.training_status import (
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD,
    get_consecutive_failures,
    get_training_status,
    record_training_result,
)


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")


def test_no_attempts_counts_zero(isolated_status):
    assert get_consecutive_failures("long_trend") == 0
    assert get_training_status()["long_trend"]["consecutive_failures"] == 0


def test_failures_accumulate(isolated_status):
    for i in range(1, 4):
        record_training_result("long_trend", success=False, error="boom")
        assert get_consecutive_failures("long_trend") == i
    assert get_training_status()["long_trend"]["consecutive_failures"] == 3


def test_success_resets_counter(isolated_status):
    record_training_result("long_trend", success=False, error="boom")
    record_training_result("long_trend", success=False, error="boom")
    record_training_result("long_trend", success=True, accuracy=0.7)
    assert get_consecutive_failures("long_trend") == 0
    # A new failure after a success starts from 1 again.
    record_training_result("long_trend", success=False, error="boom")
    assert get_consecutive_failures("long_trend") == 1


def test_counters_are_per_model(isolated_status):
    record_training_result("long_trend", success=False, error="boom")
    record_training_result("short_trend", success=True, accuracy=0.6)
    assert get_consecutive_failures("long_trend") == 1
    assert get_consecutive_failures("short_trend") == 0


def test_legacy_entry_without_counter(isolated_status):
    # Entries written before the counter existed must count from zero.
    ts.STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts.STATUS_PATH.write_text(
        '{"long_trend": {"success": false, "error": "old", "accuracy": null}}'
    )
    assert get_consecutive_failures("long_trend") == 0
    record_training_result("long_trend", success=False, error="boom")
    assert get_consecutive_failures("long_trend") == 1


def test_corrupt_file_counts_zero(isolated_status):
    ts.STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts.STATUS_PATH.write_text("{not json")
    assert get_consecutive_failures("long_trend") == 0


def test_threshold_flags_stuck(isolated_status):
    for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD):
        record_training_result("short_trend", success=False, error="boom")
    count = get_training_status()["short_trend"]["consecutive_failures"]
    assert count >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD
