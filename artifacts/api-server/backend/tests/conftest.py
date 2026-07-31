"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_recovery_history(tmp_path, monkeypatch):
    """Redirect the persisted recovery-history JSON to a temp file so tests
    never pollute the real ingestion/recovery_history.json and never see
    state left over from other tests or a running server."""
    from ingestion import recovery_history

    monkeypatch.setattr(
        recovery_history, "HISTORY_PATH", tmp_path / "recovery_history.json"
    )
    yield


@pytest.fixture(autouse=True)
def _disable_spike_quarantine_persistence():
    """
    Disable the spike-quarantine state-file for the module-level singleton so
    tests never write to the real spike_quarantine_state.json on disk and never
    pick up state left by a previous test or a running server.

    Tests that explicitly want to exercise persistence create their own
    _SpikeQuarantineTracker(state_file=...) instance with a tmp_path file.
    """
    from ingestion.ohlc_validator import _spike_tracker

    original = _spike_tracker._state_file
    _spike_tracker._state_file = ""  # "" → persistence disabled
    yield
    _spike_tracker._state_file = original


@pytest.fixture(autouse=True)
def _reset_inmemory_recovery_status():
    """Reset the module-level in-memory recovery record between tests."""
    from ingestion import pipeline

    pipeline._last_5min_recovery_status.update(
        {"last_attempt_at": None, "outcome": None, "bars_fetched": None}
    )
    yield
