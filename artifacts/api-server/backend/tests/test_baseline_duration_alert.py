"""
Baseline-mode duration alert tests.

Covers:
  1. record_baseline_mode_onset is idempotent (preserves first timestamp).
  2. clear_baseline_mode_tracking resets both fields.
  3. get_baseline_mode_days returns None when no onset is recorded.
  4. should_send_baseline_duration_alert respects threshold and sent-flag.
  5. mark_baseline_duration_alert_sent prevents a second alert in same episode.
  6. A gate-passing retrain (success=True) followed by clear_baseline_mode_tracking
     re-arms the alert for a future episode.
  7. threshold_days=0 always returns False from should_send_baseline_duration_alert.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from ml import training_status as ts
from ml.training_status import (
    clear_baseline_mode_tracking,
    get_baseline_mode_days,
    get_baseline_mode_since,
    mark_baseline_duration_alert_sent,
    record_baseline_mode_onset,
    record_training_result,
    should_send_baseline_duration_alert,
)


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")
    # Ensure a base entry exists so clear/mark have something to work with.
    record_training_result("long_trend", success=False, error="gate_fail")


# ─── onset tracking ───────────────────────────────────────────────────────────

def test_onset_sets_timestamp(isolated_status):
    record_baseline_mode_onset("long_trend")
    since = get_baseline_mode_since("long_trend")
    assert since is not None
    # Must be a parseable ISO timestamp close to now.
    dt = datetime.fromisoformat(since)
    assert abs((datetime.now(timezone.utc) - dt).total_seconds()) < 5


def test_onset_is_idempotent(isolated_status):
    """Second call must not overwrite the first timestamp."""
    record_baseline_mode_onset("long_trend")
    first = get_baseline_mode_since("long_trend")

    # Simulate time passing with a second call.
    record_baseline_mode_onset("long_trend")
    second = get_baseline_mode_since("long_trend")

    assert first == second


def test_onset_no_entry_yet(tmp_path, monkeypatch):
    """onset must succeed even when training_status.json doesn't exist yet."""
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")
    record_baseline_mode_onset("long_trend")
    since = get_baseline_mode_since("long_trend")
    assert since is not None


# ─── clear tracking ───────────────────────────────────────────────────────────

def test_clear_resets_onset_and_alert(isolated_status):
    record_baseline_mode_onset("long_trend")
    mark_baseline_duration_alert_sent("long_trend")

    clear_baseline_mode_tracking("long_trend")

    assert get_baseline_mode_since("long_trend") is None
    # After clearing, should_send fires again once threshold is passed
    # (alert_sent flag is reset).  We verify by checking the raw flag.
    data = json.loads(ts.STATUS_PATH.read_text())
    assert not data["long_trend"].get("baseline_mode_alert_sent")


def test_clear_on_missing_entry_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")
    clear_baseline_mode_tracking("long_trend")  # must not raise


# ─── days helper ──────────────────────────────────────────────────────────────

def test_days_returns_none_when_not_in_baseline(isolated_status):
    assert get_baseline_mode_days("long_trend") is None


def test_days_returns_elapsed_days(isolated_status):
    # Manually write an onset timestamp 20 days ago.
    onset = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset
    ts.STATUS_PATH.write_text(json.dumps(data))

    days = get_baseline_mode_days("long_trend")
    assert days is not None
    assert 19.9 < days < 20.1


# ─── should_send helper ───────────────────────────────────────────────────────

def test_should_send_false_when_disabled(isolated_status):
    # Write an onset 30 days ago to guarantee threshold is exceeded.
    onset = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset
    ts.STATUS_PATH.write_text(json.dumps(data))
    # threshold_days=0 disables the alert.
    assert not should_send_baseline_duration_alert("long_trend", threshold_days=0)


def test_should_send_false_before_threshold(isolated_status):
    onset = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset
    ts.STATUS_PATH.write_text(json.dumps(data))
    assert not should_send_baseline_duration_alert("long_trend", threshold_days=14)


def test_should_send_true_after_threshold(isolated_status):
    onset = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset
    ts.STATUS_PATH.write_text(json.dumps(data))
    assert should_send_baseline_duration_alert("long_trend", threshold_days=14)


def test_should_send_false_after_alert_sent(isolated_status):
    onset = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset
    ts.STATUS_PATH.write_text(json.dumps(data))

    mark_baseline_duration_alert_sent("long_trend")
    assert not should_send_baseline_duration_alert("long_trend", threshold_days=14)


def test_should_send_false_when_not_in_baseline(isolated_status):
    # No onset recorded → not in baseline mode → no alert.
    assert not should_send_baseline_duration_alert("long_trend", threshold_days=14)


# ─── re-arm after episode clears ──────────────────────────────────────────────

def test_alert_rearmed_after_clear(isolated_status):
    """After a gate-passing retrain clears the episode, a new episode can alert."""
    onset = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset
    data["long_trend"]["baseline_mode_alert_sent"] = True
    ts.STATUS_PATH.write_text(json.dumps(data))

    # Simulate gate-passing retrain.
    clear_baseline_mode_tracking("long_trend")

    # Model back in baseline (new episode):
    record_baseline_mode_onset("long_trend")
    # Not yet past threshold.
    assert not should_send_baseline_duration_alert("long_trend", threshold_days=14)

    # Now artificially age the new onset.
    onset2 = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
    data2 = json.loads(ts.STATUS_PATH.read_text())
    data2["long_trend"]["baseline_mode_since"] = onset2
    ts.STATUS_PATH.write_text(json.dumps(data2))
    assert should_send_baseline_duration_alert("long_trend", threshold_days=14)


# ─── model isolation ──────────────────────────────────────────────────────────

def test_baseline_tracking_is_per_model(isolated_status):
    record_training_result("short_trend", success=False, error="fail")
    record_baseline_mode_onset("long_trend")

    assert get_baseline_mode_since("long_trend") is not None
    assert get_baseline_mode_since("short_trend") is None


# ─── integration: baseline fields survive repeated record_training_result calls ──

def test_baseline_onset_survives_multiple_record_calls(isolated_status):
    """baseline_mode_since must not be erased by subsequent record_training_result calls.

    This is the critical regression guard: the tracking must accumulate across
    weekly retrain cycles so the elapsed-days comparison remains valid.
    """
    # Enter baseline mode and record an onset 20 days ago.
    record_baseline_mode_onset("long_trend")
    onset_20d = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset_20d
    ts.STATUS_PATH.write_text(json.dumps(data))

    # Simulate three more failed retrains (weekly cycle).
    for _ in range(3):
        record_training_result("long_trend", success=False, error="gate_fail")

    # baseline_mode_since must still carry the original 20-day-old onset.
    since = get_baseline_mode_since("long_trend")
    assert since == onset_20d, (
        f"baseline_mode_since was overwritten by record_training_result; "
        f"expected {onset_20d!r}, got {since!r}"
    )
    # Duration should still read ~20 days.
    days = get_baseline_mode_days("long_trend")
    assert days is not None and 19.9 < days < 20.1


def test_alert_eligibility_accumulates_across_retrain_cycles(isolated_status):
    """Alert must become eligible only after continuous baseline duration >= threshold.

    Simulates the real sequence:
      1. Baseline starts (onset recorded).
      2. Several retrains fail; record_training_result called each time.
      3. After enough simulated days, should_send_baseline_duration_alert fires.
    """
    threshold = 14  # days

    # Enter baseline and set onset to 5 days ago (below threshold).
    record_baseline_mode_onset("long_trend")
    onset_5d = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset_5d
    ts.STATUS_PATH.write_text(json.dumps(data))

    # Two more failed retrains — should still be below threshold.
    record_training_result("long_trend", success=False, error="gate_fail")
    record_training_result("long_trend", success=False, error="gate_fail")
    assert not should_send_baseline_duration_alert("long_trend", threshold_days=threshold)

    # Now age the onset to 16 days (past threshold).
    onset_16d = (datetime.now(timezone.utc) - timedelta(days=16)).isoformat()
    data = json.loads(ts.STATUS_PATH.read_text())
    data["long_trend"]["baseline_mode_since"] = onset_16d
    ts.STATUS_PATH.write_text(json.dumps(data))

    # One more failed retrain — onset must still be preserved.
    record_training_result("long_trend", success=False, error="gate_fail")

    since = get_baseline_mode_since("long_trend")
    assert since == onset_16d, "onset overwritten by record_training_result"

    # Alert should now be eligible.
    assert should_send_baseline_duration_alert("long_trend", threshold_days=threshold)

    # Mark it sent; eligibility must not re-fire on the next cycle.
    mark_baseline_duration_alert_sent("long_trend")
    record_training_result("long_trend", success=False, error="gate_fail")
    assert not should_send_baseline_duration_alert("long_trend", threshold_days=threshold)


def test_alert_sent_flag_survives_record_training_result(isolated_status):
    """baseline_mode_alert_sent must not be reset by record_training_result."""
    record_baseline_mode_onset("long_trend")
    mark_baseline_duration_alert_sent("long_trend")

    # Subsequent failed retrain must carry the sent flag forward.
    record_training_result("long_trend", success=False, error="gate_fail")

    data = json.loads(ts.STATUS_PATH.read_text())
    assert data["long_trend"].get("baseline_mode_alert_sent") is True
