"""A retrain whose purged walk-forward evaluation could not run (too few
rows) reports train accuracy (accuracy_metric='train'). That condition must
be visible in /api/healthz so operators know the headline accuracy is not an
honest OOS metric.
"""

import pytest

from ml import training_status as ts


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")


class TestHealthzWalkForwardSkipVisibility:
    @pytest.mark.asyncio
    async def test_healthz_flags_train_accuracy_metric(
        self, isolated_status, monkeypatch
    ):
        ts.record_training_result(
            "short_trend",
            success=True,
            accuracy=0.986,
            accuracy_metric="train",
        )
        ts.record_training_result(
            "long_trend",
            success=True,
            accuracy=0.71,
            accuracy_metric="purged_walk_forward_oos",
        )

        from httpx import AsyncClient, ASGITransport
        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")
        assert resp.status_code == 200
        body = resp.json()

        short = body["models"]["short_trend"]
        assert short["last_training_accuracy_metric"] == "train"
        assert short["walk_forward_evaluation_skipped"] is True
        assert any(
            a.startswith("short_trend") and "walk-forward evaluation" in a
            for a in body["alerts"]
        )

        long = body["models"]["long_trend"]
        assert long["last_training_accuracy_metric"] == "purged_walk_forward_oos"
        assert long["walk_forward_evaluation_skipped"] is False
        assert not any(
            a.startswith("long_trend") and "walk-forward evaluation" in a
            for a in body["alerts"]
        )

    @pytest.mark.asyncio
    async def test_healthz_no_flag_when_no_training_yet(
        self, isolated_status, monkeypatch
    ):
        from httpx import AsyncClient, ASGITransport
        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")
        assert resp.status_code == 200
        body = resp.json()
        for name in ("short_trend", "long_trend"):
            assert body["models"][name]["walk_forward_evaluation_skipped"] is False
            assert body["models"][name]["last_training_accuracy_metric"] is None
