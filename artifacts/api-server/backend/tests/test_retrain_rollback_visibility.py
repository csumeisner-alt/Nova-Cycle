"""A rolled-back retrain must be visible to operators, not just logged.

Covers:
  1. training_status records rolled_back=True and surfaces it via
     get_training_status().
  2. /api/healthz reports last_retrain_outcome="rolled_back" with the
     attempted (discarded) accuracy alongside the active model's accuracy,
     and emits a human-readable rollback alert.
"""

import pytest

from ml import training_status as ts


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")


class TestTrainingStatusRolledBack:
    def test_rolled_back_recorded_and_surfaced(self, isolated_status):
        ts.record_training_result("long_trend", success=True, accuracy=0.71)
        ts.record_training_result(
            "long_trend",
            success=False,
            error="accuracy regression: 0.52 vs last-good 0.71",
            accuracy=0.52,
            rolled_back=True,
        )
        status = ts.get_training_status()["long_trend"]
        assert status["success"] is False
        assert status["rolled_back"] is True
        assert status["accuracy"] == 0.52
        assert status["last_success_accuracy"] == 0.71

    def test_success_clears_rolled_back(self, isolated_status):
        ts.record_training_result(
            "long_trend", success=False, error="regressed", rolled_back=True
        )
        ts.record_training_result("long_trend", success=True, accuracy=0.72)
        status = ts.get_training_status()["long_trend"]
        assert status["rolled_back"] is False
        assert status["success"] is True

    def test_plain_failure_not_marked_rolled_back(self, isolated_status):
        ts.record_training_result("short_trend", success=False, error="no data")
        status = ts.get_training_status()["short_trend"]
        assert status["success"] is False
        assert status["rolled_back"] is False

    def test_missing_entry_defaults(self, isolated_status):
        status = ts.get_training_status()["short_trend"]
        assert status["rolled_back"] is False
        assert status["last_success_accuracy"] is None


class TestHealthzRollbackVisibility:
    @pytest.mark.asyncio
    async def test_healthz_reports_rolled_back_outcome(
        self, isolated_status, monkeypatch
    ):
        ts.record_training_result("long_trend", success=True, accuracy=0.71)
        ts.record_training_result(
            "long_trend",
            success=False,
            error="accuracy regression: new 0.5200 vs last-good 0.7100",
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

        long = body["models"]["long_trend"]
        assert long["last_retrain_outcome"] == "rolled_back"
        assert long["last_retrain_rolled_back"] is True
        assert long["last_retrain_attempted_accuracy"] == 0.52
        # Active accuracy must reflect the last-good model, not the discarded one.
        assert long["active_model_accuracy"] != 0.52
        assert body["status"] == "degraded"
        assert any(
            "rolled back" in a and a.startswith("long_trend") for a in body["alerts"]
        )

        short = body["models"]["short_trend"]
        assert short["last_retrain_outcome"] == "success"
        assert short["last_retrain_rolled_back"] is False
