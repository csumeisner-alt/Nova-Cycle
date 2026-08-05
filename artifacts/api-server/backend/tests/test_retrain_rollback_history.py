"""Rollback history grows correctly across multiple simulated rollbacks.

Covers:
  1. record_rollback_event appends events with the correct fields.
  2. History grows across multiple rollbacks (different models, success/fail).
  3. record_training_result(rolled_back=True) automatically appends a history event.
  4. get_rollback_history returns events newest-first and honours last_n.
  5. ROLLBACK_HISTORY_MAX_EVENTS cap is respected (oldest entries are dropped).
  6. /api/healthz surfaces rollback_history as a list.
"""

import pytest

from ml import training_status as ts


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    """Redirect both status JSON and rollback-history JSON to tmp_path."""
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")
    monkeypatch.setattr(ts, "ROLLBACK_HISTORY_PATH", tmp_path / "rollback_history.json")


class TestRecordRollbackEvent:
    def test_first_event_is_stored(self, isolated_history):
        ts.record_rollback_event("long_trend", reason="accuracy regression", restore_succeeded=True)
        history = ts.get_rollback_history()
        assert len(history) == 1
        ev = history[0]
        assert ev["model_name"] == "long_trend"
        assert ev["reason"] == "accuracy regression"
        assert ev["restore_succeeded"] is True
        assert ev["timestamp"]  # non-empty ISO string

    def test_multiple_events_accumulate(self, isolated_history):
        ts.record_rollback_event("long_trend", reason="regression #1", restore_succeeded=True)
        ts.record_rollback_event("short_trend", reason="degenerate", restore_succeeded=True)
        ts.record_rollback_event("long_trend", reason="no backup", restore_succeeded=False)

        history = ts.get_rollback_history()
        assert len(history) == 3
        # Newest first
        assert history[0]["reason"] == "no backup"
        assert history[0]["restore_succeeded"] is False
        assert history[1]["model_name"] == "short_trend"
        assert history[2]["reason"] == "regression #1"

    def test_restore_succeeded_false_recorded(self, isolated_history):
        ts.record_rollback_event("short_trend", reason="no_backup_available", restore_succeeded=False)
        history = ts.get_rollback_history()
        assert len(history) == 1
        assert history[0]["restore_succeeded"] is False

    def test_last_n_limits_returned_events(self, isolated_history):
        for i in range(10):
            ts.record_rollback_event("long_trend", reason=f"event {i}", restore_succeeded=True)

        history = ts.get_rollback_history(last_n=3)
        assert len(history) == 3
        # Should be the last 3 events, newest first
        assert history[0]["reason"] == "event 9"
        assert history[1]["reason"] == "event 8"
        assert history[2]["reason"] == "event 7"

    def test_history_cap_drops_oldest(self, isolated_history, monkeypatch):
        monkeypatch.setattr(ts, "ROLLBACK_HISTORY_MAX_EVENTS", 5)
        for i in range(8):
            ts.record_rollback_event("long_trend", reason=f"event {i}", restore_succeeded=True)

        # Only the 5 most recent are kept
        raw = ts._load_rollback_history_raw()
        assert len(raw) == 5
        reasons = [e["reason"] for e in raw]
        assert "event 0" not in reasons
        assert "event 7" in reasons

    def test_reason_truncated_to_500_chars(self, isolated_history):
        long_reason = "x" * 600
        ts.record_rollback_event("long_trend", reason=long_reason, restore_succeeded=True)
        history = ts.get_rollback_history()
        assert len(history[0]["reason"]) == 500

    def test_none_reason_stored_as_none(self, isolated_history):
        ts.record_rollback_event("long_trend", reason=None, restore_succeeded=False)
        history = ts.get_rollback_history()
        assert history[0]["reason"] is None


class TestRecordTrainingResultAutoAppends:
    """record_training_result(rolled_back=True) must auto-append a history event."""

    def test_rolled_back_failure_appends_event(self, isolated_history):
        ts.record_training_result(
            "long_trend",
            success=False,
            error="accuracy regression: new 0.42 vs last-good 0.70",
            accuracy=0.42,
            rolled_back=True,
        )
        history = ts.get_rollback_history()
        assert len(history) == 1
        ev = history[0]
        assert ev["model_name"] == "long_trend"
        assert "regression" in ev["reason"]
        assert ev["restore_succeeded"] is True

    def test_plain_failure_does_not_append(self, isolated_history):
        ts.record_training_result(
            "short_trend",
            success=False,
            error="insufficient data",
            rolled_back=False,
        )
        assert ts.get_rollback_history() == []

    def test_success_does_not_append(self, isolated_history):
        ts.record_training_result("long_trend", success=True, accuracy=0.70)
        assert ts.get_rollback_history() == []

    def test_multiple_rollbacks_grow_history(self, isolated_history):
        """Simulate three sequential rollbacks across two models."""
        ts.record_training_result("long_trend", success=True, accuracy=0.70)
        ts.record_training_result(
            "long_trend", success=False, error="regression rollback #1",
            accuracy=0.50, rolled_back=True,
        )
        ts.record_training_result(
            "short_trend", success=False, error="degenerate rollback",
            accuracy=0.55, rolled_back=True,
        )
        ts.record_training_result(
            "long_trend", success=False, error="regression rollback #2",
            accuracy=0.48, rolled_back=True,
        )

        history = ts.get_rollback_history()
        assert len(history) == 3
        # Newest first
        assert "rollback #2" in history[0]["reason"]
        assert history[1]["model_name"] == "short_trend"
        assert "rollback #1" in history[2]["reason"]
        # All should have restore_succeeded=True
        assert all(ev["restore_succeeded"] for ev in history)


class TestHealthzRollbackHistory:
    @pytest.mark.asyncio
    async def test_healthz_surfaces_rollback_history(self, isolated_history, monkeypatch):
        """The /api/healthz response must include rollback_history as a list."""
        ts.record_training_result(
            "long_trend", success=True, accuracy=0.71,
            accuracy_metric="purged_walk_forward_oos",
        )
        ts.record_training_result(
            "long_trend",
            success=False,
            error="accuracy regression: new 0.52 vs last-good 0.71",
            accuracy=0.52,
            rolled_back=True,
        )
        ts.record_training_result("short_trend", success=True, accuracy=0.66)

        from httpx import AsyncClient, ASGITransport
        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()

        assert "rollback_history" in body
        assert isinstance(body["rollback_history"], list)
        assert len(body["rollback_history"]) >= 1
        ev = body["rollback_history"][0]  # most recent first
        assert ev["model_name"] == "long_trend"
        assert ev["restore_succeeded"] is True
        assert "regression" in ev["reason"]
        assert "timestamp" in ev
