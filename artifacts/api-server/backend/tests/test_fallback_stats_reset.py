"""Tests for the operator reset of persisted ML-fallback history.

Covers the fallback_stats reset helper (persisted file behaviour) and the
authenticated POST /api/admin/reset_fallback_stats endpoint, including that
/api/healthz-visible counters (in-memory + persisted) are cleared immediately.
"""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import ml.fallback_stats as fs
from config import settings
from main import app
import routers.predictions as predictions


@pytest.fixture
def stats_file(tmp_path, monkeypatch):
    path = tmp_path / "ml_fallback_stats.json"
    monkeypatch.setattr(fs, "STATS_PATH", path)
    return path


# ---------------------------------------------------------------------------
# Unit: fallback_stats helpers
# ---------------------------------------------------------------------------
def test_reset_clears_persisted_totals(stats_file):
    fs.record_fallback("long_trend", "boom")
    fs.record_fallback("long_trend", "boom again")
    fs.record_fallback("short_trend", "other")
    assert fs.get_persisted_fallback_stats()["long_trend"]["total_count"] == 2

    previous = fs.reset_fallback_stats()
    assert previous["long_trend"]["total_count"] == 2
    assert previous["short_trend"]["total_count"] == 1

    after = fs.get_persisted_fallback_stats()
    for name in ("long_trend", "short_trend"):
        assert after[name] == {"total_count": 0, "last_at": None, "last_reason": None}

    # Audit trail is kept in the file and surfaced via get_last_reset_at
    raw = json.loads(stats_file.read_text())
    assert raw["_last_reset"]["previous"]["long_trend"]["total_count"] == 2
    assert fs.get_last_reset_at() == raw["_last_reset"]["at"]


def test_counters_resume_after_reset(stats_file):
    fs.record_fallback("long_trend", "boom")
    fs.reset_fallback_stats()
    fs.record_fallback("long_trend", "new problem")
    stats = fs.get_persisted_fallback_stats()
    assert stats["long_trend"]["total_count"] == 1
    assert stats["long_trend"]["last_reason"] == "new problem"
    assert fs.get_last_reset_at() is not None


def test_get_last_reset_at_none_when_never_reset(stats_file):
    assert fs.get_last_reset_at() is None


# ---------------------------------------------------------------------------
# Endpoint: POST /api/admin/reset_fallback_stats
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_reset_endpoint_disabled_without_token(client, stats_file, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "")
    monkeypatch.setattr(settings, "SESSION_SECRET", "")
    resp = await client.post("/api/admin/reset_fallback_stats",
                             headers={"X-Admin-Token": "anything"})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_reset_endpoint_rejects_bad_token(client, stats_file, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "correct-token")
    resp = await client.post("/api/admin/reset_fallback_stats")
    assert resp.status_code == 403
    resp = await client.post("/api/admin/reset_fallback_stats",
                             headers={"X-Admin-Token": "wrong"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_reset_endpoint_clears_persisted_and_in_memory(client, stats_file, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "correct-token")
    fs.record_fallback("long_trend", "boom")
    predictions._ml_fallback_stats["long_trend"].update(
        {"count": 3, "last_at": "2026-07-25T00:00:00", "last_reason": "boom"}
    )

    resp = await client.post("/api/admin/reset_fallback_stats",
                             headers={"X-Admin-Token": "correct-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["previous_persisted"]["long_trend"]["total_count"] == 1
    assert body["previous_in_memory"]["long_trend"]["count"] == 3

    assert fs.get_persisted_fallback_stats()["long_trend"]["total_count"] == 0
    assert predictions._ml_fallback_stats["long_trend"] == {
        "count": 0, "last_at": None, "last_reason": None
    }


@pytest.mark.asyncio
async def test_reset_endpoint_falls_back_to_session_secret(client, stats_file, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "")
    monkeypatch.setattr(settings, "SESSION_SECRET", "sess-secret")
    resp = await client.post("/api/admin/reset_fallback_stats",
                             headers={"X-Admin-Token": "sess-secret"})
    assert resp.status_code == 200
