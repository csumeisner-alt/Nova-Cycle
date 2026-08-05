"""
/api/healthz broader_context_ablation field — surface & safety tests.

Verifies three contracts:
  1. Field is null when no ablation report file exists.
  2. Field carries the full report payload when the file exists.
  3. Field stays null (never 500) when the file contains corrupt JSON.
"""

import json
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

# Absolute path the /api/healthz handler reads the ablation report from.
_ABLATION_PATH = (
    Path(__file__).resolve().parents[1] / "ml" / "models" / "ablation_broader_context.json"
)


@pytest.fixture
def ablation_report():
    """Minimal valid ablation report matching the script's output schema."""
    return {
        "ablation": "broader_context_features",
        "run_timestamp_utc": "2026-07-01T12:00:00",
        "data_source": "yfinance",
        "n_labeled_rows": 400,
        "majority_baseline": 0.55,
        "LONG_MIN_OOS_ACCURACY_LIFT": 0.02,
        "baseline_19feat": {
            "n_features": 19,
            "oos_accuracy": 0.60,
            "oos_lift_vs_majority": 0.05,
            "oos_balanced_accuracy": 0.59,
            "folds": 5,
        },
        "candidate_27feat": {
            "n_features": 27,
            "oos_accuracy": 0.63,
            "oos_lift_vs_majority": 0.08,
            "oos_balanced_accuracy": 0.61,
            "folds": 5,
            "context_feature_importances": {},
        },
        "accuracy_delta_27_minus_19": 0.03,
        "balanced_accuracy_delta": 0.02,
        "passes_promotion_gate": True,
        "promotion_gate_description": (
            "27-feat OOS lift >= LONG_MIN_OOS_ACCURACY_LIFT "
            "AND 27-feat OOS accuracy > 19-feat OOS accuracy"
        ),
        "recommendation": (
            "Enable LONG_BROADER_CONTEXT_ENABLED=True after a gate-passing retrain."
        ),
        "promotion_record": None,
    }


@pytest.fixture(autouse=True)
def _remove_ablation_file():
    """
    Guarantee the ablation report file is absent before each test and
    restored to its original state (absent) after.  Tests that need the
    file present write it themselves; this fixture cleans up after them.
    """
    existed_before = _ABLATION_PATH.exists()
    original_content = _ABLATION_PATH.read_bytes() if existed_before else None
    # Remove before test so each starts from a known-clean state.
    if _ABLATION_PATH.exists():
        _ABLATION_PATH.unlink()
    yield
    # Restore.
    if original_content is not None:
        _ABLATION_PATH.write_bytes(original_content)
    elif _ABLATION_PATH.exists():
        _ABLATION_PATH.unlink()


class TestHealthzBroaderContextAblation:

    @pytest.mark.asyncio
    async def test_field_is_null_when_no_report_exists(self):
        """broader_context_ablation is null when the ablation file is absent."""
        assert not _ABLATION_PATH.exists(), "Fixture should have removed the file"

        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()
        assert "broader_context_ablation" in body
        assert body["broader_context_ablation"] is None

    @pytest.mark.asyncio
    async def test_field_carries_report_when_file_exists(self, ablation_report):
        """broader_context_ablation mirrors the JSON file exactly when present."""
        _ABLATION_PATH.write_text(json.dumps(ablation_report))

        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()
        result = body["broader_context_ablation"]
        assert result is not None

        # Core fields required by the task spec.
        assert result["run_timestamp_utc"] == ablation_report["run_timestamp_utc"]
        assert result["baseline_19feat"]["oos_accuracy"] == pytest.approx(
            ablation_report["baseline_19feat"]["oos_accuracy"]
        )
        assert result["candidate_27feat"]["oos_accuracy"] == pytest.approx(
            ablation_report["candidate_27feat"]["oos_accuracy"]
        )
        assert result["accuracy_delta_27_minus_19"] == pytest.approx(
            ablation_report["accuracy_delta_27_minus_19"]
        )
        assert result["passes_promotion_gate"] == ablation_report["passes_promotion_gate"]

    @pytest.mark.asyncio
    async def test_field_is_null_on_corrupt_json(self):
        """broader_context_ablation is null (no 500) when the file contains corrupt JSON."""
        _ABLATION_PATH.write_text("{ INVALID JSON }")

        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()
        assert "broader_context_ablation" in body
        assert body["broader_context_ablation"] is None
