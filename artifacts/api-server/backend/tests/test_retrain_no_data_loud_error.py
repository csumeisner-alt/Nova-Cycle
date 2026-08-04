"""
Tests that the weekly retrain surfaces a loud error (logger.error + health
status) when no daily VOO candle data is available, and that the database
module warns when a non-SQLite DATABASE_URL env var is present but ignored.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

import ml.training_status as ts
from ml.trainer import ModelTrainer
from ml.training_status import get_training_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal async session stub."""

    def __init__(self):
        self.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
        self.add = MagicMock()
        self.flush = AsyncMock()
        self.commit = AsyncMock()
        self.rollback = AsyncMock()


@pytest.fixture()
def isolated_status(tmp_path, monkeypatch):
    """Redirect training-status persistence to a temp file."""
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")


# ---------------------------------------------------------------------------
# Test 1 — empty daily candles → logger.error + health status recorded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_daily_voo_logs_error_and_records_failure(
    isolated_status, caplog
):
    """When _load_daily_voo returns an empty DataFrame, run_initial_training
    must emit logger.error (not just warning) and record a failed training
    result so the /healthz endpoint surfaces the problem."""

    trainer = ModelTrainer.__new__(ModelTrainer)
    trainer.indicators = MagicMock()
    trainer.long_model = MagicMock()
    trainer.short_model = MagicMock()

    # _load_daily_voo is a staticmethod; patch at module level so the
    # coroutine signature matches (db_session only, no self).
    async def _empty_daily(*args, **kwargs):
        return pd.DataFrame()

    async def _noop_stuck_alert(*args, **kwargs):
        pass

    with patch.object(ModelTrainer, "_load_daily_voo", staticmethod(_empty_daily)), \
         patch.object(ModelTrainer, "_maybe_send_stuck_alert", _noop_stuck_alert):
        with caplog.at_level(logging.ERROR, logger="ml.trainer"):
            await trainer.run_initial_training(_FakeSession())

    # Must emit at least one ERROR-level log message mentioning the problem
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, (
        "Expected an ERROR-level log when daily VOO data is missing, got none"
    )
    assert any(
        "no_data" in r.message or "candle table is empty" in r.message
        or "No daily VOO" in r.message
        for r in error_records
    ), f"Error message did not mention missing data: {[r.message for r in error_records]}"

    # Health status must record a failure for both models
    status = get_training_status()
    assert status["long_trend"]["success"] is False, (
        "long_trend training status should be False when data is absent"
    )
    assert status["short_trend"]["success"] is False, (
        "short_trend training status should be False when data is absent"
    )
    assert "No daily VOO data" in (status["long_trend"]["error"] or "")
    assert "No daily VOO data" in (status["short_trend"]["error"] or "")


# ---------------------------------------------------------------------------
# Test 2 — health status is degraded after no-data retrain
# ---------------------------------------------------------------------------

def test_no_data_failure_increments_consecutive_failures(isolated_status):
    """Each no-data abort must increment the consecutive_failures counter so
    the training-stuck alert fires after the threshold is reached."""
    from ml.training_status import record_training_result, get_consecutive_failures

    for i in range(1, 4):
        record_training_result(
            "long_trend", success=False, error="No daily VOO data available"
        )
        assert get_consecutive_failures("long_trend") == i


# ---------------------------------------------------------------------------
# Test 3 — DATABASE_URL warning is emitted when a non-SQLite URL is set
# ---------------------------------------------------------------------------

def test_database_url_ignored_warning_logged(caplog, monkeypatch):
    """When DATABASE_URL is set to a PostgreSQL URL, db.py must emit a
    WARNING so operators see the misleading env var in production logs.
    The warning must NOT include credentials (password or user info)."""

    import sys

    pg_url = "postgresql+asyncpg://myuser:s3cr3tpassword@some-host/novacycle"
    monkeypatch.setenv("DATABASE_URL", pg_url)

    # Force db.py to re-execute its module-level warning logic by removing
    # it from sys.modules and re-importing.
    sys.modules.pop("database.db", None)

    with caplog.at_level(logging.WARNING, logger="database.db"):
        import database.db  # noqa: F401 — triggers the module-level check

    warning_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "db_url_ignored" in r.message
    ]
    assert warning_records, (
        "Expected a WARNING about DATABASE_URL being ignored, got none. "
        f"All records: {[(r.levelno, r.message) for r in caplog.records]}"
    )

    # Security: credentials must never appear in the log output
    for record in warning_records:
        msg = record.message
        assert "s3cr3tpassword" not in msg, (
            f"Password leaked into log message: {msg!r}"
        )
        assert "myuser" not in msg, (
            f"Username leaked into log message: {msg!r}"
        )
        # The host/scheme should appear so operators know what was ignored
        assert "some-host" in msg or "postgresql" in msg, (
            f"Expected scheme or host in log message to identify the URL: {msg!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — no warning when DATABASE_URL is absent
# ---------------------------------------------------------------------------

def test_no_warning_when_database_url_not_set(caplog, monkeypatch):
    """When DATABASE_URL is not set (normal SQLite-only deployment), no
    warning about an ignored env var should be emitted."""
    import sys

    monkeypatch.delenv("DATABASE_URL", raising=False)
    sys.modules.pop("database.db", None)

    with caplog.at_level(logging.WARNING, logger="database.db"):
        import database.db  # noqa: F401

    ignored_warnings = [
        r for r in caplog.records if "db_url_ignored" in r.message
    ]
    assert not ignored_warnings, (
        "Should not warn about DATABASE_URL when it is not set"
    )
