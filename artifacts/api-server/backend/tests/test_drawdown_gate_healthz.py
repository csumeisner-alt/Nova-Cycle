"""
/api/healthz drawdown_gate field — safety tests.

Verifies two contracts:
  1. Field is null when no dry-run gate file exists (file absent).
  2. Field is null (logged error, no 500) when the file contains invalid JSON.
"""

import json
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

# Absolute path the /api/healthz handler reads the drawdown gate report from.
_GATE_PATH = (
    Path(__file__).resolve().parents[1] / "ml" / "models" / "drawdown_dry_run.json"
)

# Minimal valid gate report matching the dry-run script's output schema.
_VALID_REPORT = {
    "run_timestamp_utc": "2026-07-15T10:00:00",
    "data_source": "yfinance",
    "total_configs_evaluated": 8,
    "configs_passing_gate": 0,
    "promotion_gate_description": "PR-AUC_lift>=2 AND precision_lift>=2 (no auto-promote)",
    "best_result": {
        "label": "h5_dd0.05_xgb",
        "horizon": 5,
        "drawdown_thresh": 0.05,
        "model": "xgb",
        "passes_promotion_gate": False,
        "pr_auc_lift_vs_prevalence": 1.2,
        "precision_lift_vs_base_rate": 1.1,
        "avoided_drawdown_recall": 0.40,
        "positive_rate": 0.08,
    },
    "passing_results": [],
}


@pytest.fixture(autouse=True)
def _isolate_gate_file():
    """
    Guarantee the gate file is absent before each test and restored after.
    Tests that need the file present write it themselves; this fixture cleans
    up after them so no test pollutes the real ml/models/ directory.
    """
    existed_before = _GATE_PATH.exists()
    original_content = _GATE_PATH.read_bytes() if existed_before else None
    if _GATE_PATH.exists():
        _GATE_PATH.unlink()
    yield
    if original_content is not None:
        _GATE_PATH.write_bytes(original_content)
    elif _GATE_PATH.exists():
        _GATE_PATH.unlink()


class TestHealthzDrawdownGate:

    @pytest.mark.asyncio
    async def test_field_is_null_when_file_absent(self):
        """drawdown_gate is null (not a 500) when the json file does not exist."""
        assert not _GATE_PATH.exists(), "Fixture should have removed the file"

        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()
        assert "drawdown_gate" in body, "drawdown_gate key must always be present"
        assert body["drawdown_gate"] is None

    @pytest.mark.asyncio
    async def test_field_is_null_on_invalid_json(self, caplog):
        """
        drawdown_gate is null (no 500) when the file contains syntactically
        invalid JSON (e.g. disk full during write, partial flush).
        The backend must log an error so the operator knows the file is bad.
        """
        import logging

        _GATE_PATH.write_text("{ NOT VALID JSON !!!}")

        from main import app

        transport = ASGITransport(app=app)
        with caplog.at_level(logging.ERROR):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()
        assert "drawdown_gate" in body
        assert body["drawdown_gate"] is None, (
            "Corrupt gate file should return null — no 500 and no sentinel object"
        )
        # The backend must log the parse error so operators know the file is bad.
        assert any(
            "drawdown" in record.message.lower() or "gate" in record.message.lower()
            for record in caplog.records
            if record.levelno >= logging.ERROR
        ), "Expected an ERROR log mentioning the drawdown gate load failure"

    @pytest.mark.asyncio
    async def test_field_is_null_on_truncated_json(self):
        """
        drawdown_gate is null (no 500) when the file is truncated mid-write
        (simulating a race between the dry-run script and the healthz reader).
        """
        partial = '{"run_timestamp_utc": "2026-07-15T10:00:00", "total_configs_'
        _GATE_PATH.write_text(partial)

        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()
        assert "drawdown_gate" in body
        assert body["drawdown_gate"] is None, (
            "Truncated gate file should return null, not a 500"
        )

    @pytest.mark.asyncio
    async def test_field_carries_report_when_file_exists(self):
        """drawdown_gate mirrors the JSON file exactly when the file is valid."""
        _GATE_PATH.write_text(json.dumps(_VALID_REPORT))

        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()
        result = body["drawdown_gate"]
        assert result is not None
        assert result["run_timestamp_utc"] == _VALID_REPORT["run_timestamp_utc"]
        assert result["total_configs_evaluated"] == _VALID_REPORT["total_configs_evaluated"]
        assert result["configs_passing_gate"] == _VALID_REPORT["configs_passing_gate"]
        best = result.get("best_result") or {}
        assert best.get("label") == _VALID_REPORT["best_result"]["label"]
