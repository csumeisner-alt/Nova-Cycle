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
def _reset_inmemory_recovery_status():
    """Reset the module-level in-memory recovery record between tests."""
    from ingestion import pipeline

    pipeline._last_5min_recovery_status.update(
        {"last_attempt_at": None, "outcome": None, "bars_fetched": None}
    )
    yield
