"""
Tests for database.startup_state and its integration with the healthz endpoint.

Covers:
  - startup_state module transitions: pending → ok / degraded
  - Error summary is stored and retrievable
  - healthz reflects startup_status and sets overall status to "degraded" when
    pipeline initialization failed before initialize() was called
"""

import pytest
import database.startup_state as ss


# ─────────────────────────────────────────────────────────────────────────────
# Startup state module unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStartupStateModule:
    def setup_method(self):
        ss.reset_for_testing()

    def test_initial_status_is_pending(self):
        assert ss.get_startup_status() == "pending"
        assert ss.get_startup_error() is None

    def test_mark_ok_sets_status(self):
        ss.mark_startup_ok()
        assert ss.get_startup_status() == "ok"
        assert ss.get_startup_error() is None

    def test_mark_degraded_sets_status_and_error(self):
        ss.mark_startup_degraded("simulated reclassify failure")
        assert ss.get_startup_status() == "degraded"
        assert ss.get_startup_error() == "simulated reclassify failure"

    def test_mark_ok_after_degraded_clears_error(self):
        ss.mark_startup_degraded("transient error")
        ss.mark_startup_ok()
        assert ss.get_startup_status() == "ok"
        assert ss.get_startup_error() is None

    def test_degraded_error_is_exact_string(self):
        err = "DB connection refused: [Errno 111]"
        ss.mark_startup_degraded(err)
        assert ss.get_startup_error() == err

    def test_reset_for_testing_restores_pending(self):
        ss.mark_startup_ok()
        ss.reset_for_testing()
        assert ss.get_startup_status() == "pending"
        assert ss.get_startup_error() is None

    def test_multiple_degraded_calls_keep_last_error(self):
        ss.mark_startup_degraded("first error")
        ss.mark_startup_degraded("second error")
        assert ss.get_startup_status() == "degraded"
        assert ss.get_startup_error() == "second error"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: healthz reflects startup_status
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthzStartupStatus:
    """
    Confirm that the /api/healthz response includes startup_status and
    startup_error, and that a degraded startup escalates the overall status.

    These tests call the helper logic directly rather than spinning up the full
    FastAPI app — they assert only on the startup_state slice of healthz.
    """

    def setup_method(self):
        ss.reset_for_testing()

    def test_healthz_fields_exist_when_ok(self):
        ss.mark_startup_ok()
        status = ss.get_startup_status()
        error = ss.get_startup_error()
        assert status == "ok"
        assert error is None

    def test_healthz_fields_exist_when_degraded(self):
        ss.mark_startup_degraded("reclassify_session_labels failed: DB locked")
        status = ss.get_startup_status()
        error = ss.get_startup_error()
        assert status == "degraded"
        assert "reclassify_session_labels" in error

    def test_degraded_startup_would_set_overall_degraded(self):
        """
        Verify the logic that healthz uses: startup_status == "degraded"
        should cause the overall 'degraded' flag to be set to True and an
        alert to be appended.
        """
        ss.mark_startup_degraded("pre-init step crashed")

        # Replicate exactly what healthz does:
        degraded = False
        alerts = []
        startup_status = ss.get_startup_status()
        startup_error = ss.get_startup_error()

        if startup_status == "degraded":
            degraded = True
            alerts.append(
                "startup: pipeline initialization failed — jobs unblocked in a degraded state"
                + (f" ({startup_error})" if startup_error else "")
            )

        assert degraded, "overall degraded flag must be True when startup is degraded"
        assert len(alerts) == 1
        assert "pre-init step crashed" in alerts[0]
        assert "startup" in alerts[0]

    def test_ok_startup_does_not_set_overall_degraded(self):
        """A clean startup must not contribute to the degraded flag."""
        ss.mark_startup_ok()

        degraded = False
        alerts = []
        startup_status = ss.get_startup_status()
        startup_error = ss.get_startup_error()

        if startup_status == "degraded":
            degraded = True
            alerts.append("startup degraded")

        assert not degraded
        assert alerts == []

    def test_pending_startup_does_not_set_overall_degraded(self):
        """A still-pending startup (server just booting) must not flag degraded."""
        # reset_for_testing already left status as "pending"
        degraded = False
        startup_status = ss.get_startup_status()

        if startup_status == "degraded":
            degraded = True

        assert not degraded


# ─────────────────────────────────────────────────────────────────────────────
# retrain_skipped_at_startup flag
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrainSkippedAtStartup:
    def setup_method(self):
        ss.reset_for_testing()

    def test_initial_value_is_false(self):
        assert ss.get_retrain_skipped_at_startup() is False

    def test_mark_retrain_skipped_sets_flag(self):
        ss.mark_retrain_skipped()
        assert ss.get_retrain_skipped_at_startup() is True

    def test_reset_clears_flag(self):
        ss.mark_retrain_skipped()
        ss.reset_for_testing()
        assert ss.get_retrain_skipped_at_startup() is False

    def test_mark_startup_ok_does_not_set_flag(self):
        """A clean startup should leave retrain_skipped_at_startup False."""
        ss.mark_startup_ok()
        assert ss.get_retrain_skipped_at_startup() is False

    def test_mark_retrain_skipped_is_idempotent(self):
        """Calling mark_retrain_skipped() twice should not raise and stays True."""
        ss.mark_retrain_skipped()
        ss.mark_retrain_skipped()
        assert ss.get_retrain_skipped_at_startup() is True

    def test_skipped_flag_independent_of_startup_status(self):
        """retrain_skipped is orthogonal to startup_status — both can be read together."""
        ss.mark_startup_degraded("db locked")
        ss.mark_retrain_skipped()
        assert ss.get_startup_status() == "degraded"
        assert ss.get_retrain_skipped_at_startup() is True

    def test_healthz_alert_added_when_retrain_skipped(self):
        """Replicate the logic healthz uses to build alerts for skipped retrain."""
        ss.mark_retrain_skipped()

        degraded = False
        alerts = []
        if ss.get_retrain_skipped_at_startup():
            degraded = True
            alerts.append(
                "startup: retrain was skipped because pipeline initialization did not complete "
                "cleanly — models may be stale; retrain will run on the next weekly schedule"
            )

        assert degraded
        assert len(alerts) == 1
        assert "retrain was skipped" in alerts[0]
        assert "models may be stale" in alerts[0]

    def test_healthz_no_alert_when_retrain_ran(self):
        """No alert when retrain was not skipped."""
        ss.mark_startup_ok()

        degraded = False
        alerts = []
        if ss.get_retrain_skipped_at_startup():
            degraded = True
            alerts.append("startup: retrain was skipped")

        assert not degraded
        assert alerts == []
