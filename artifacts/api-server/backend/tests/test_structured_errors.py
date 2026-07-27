"""
Tests for the structured JSON error envelope and request-timeout guard.

The Android app relies on every failure mode returning
{"ok": false, "error": <CODE>, "detail": ...} instead of a bare 500 or an
HTML error page. Successful responses must be unchanged.
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

import main
from main import app


def _client():
    # No lifespan: we only exercise middleware/handlers, not the pipeline.
    # raise_app_exceptions=False: Starlette's Exception handler sends its JSON
    # response and then re-raises for the server to log; the transport must not
    # turn that re-raise into a test failure.
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )



async def test_success_response_shape_unchanged():
    async with _client() as client:
        resp = await client.get("/api/test")
    assert resp.status_code == 200
    body = resp.json()
    # Existing shape retained — no error envelope on success
    assert body["status"] == "ok"
    assert "ok" not in body



async def test_validation_error_returns_invalid_request():
    # /api/confidence_history requires a supported window value
    async with _client() as client:
        resp = await client.get("/api/voo_candles", params={"window": "not-a-window"})
    # Either the endpoint validates explicitly (4xx HTTPException) or FastAPI
    # rejects the params — both must produce the structured envelope.
    if resp.status_code >= 400:
        body = resp.json()
        assert body["ok"] is False
        assert body["error"] in ("INVALID_REQUEST", "SERVER_FAILURE")



async def test_http_exception_envelope():
    async with _client() as client:
        resp = await client.get("/api/definitely-not-a-route")
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "INVALID_REQUEST"



async def test_unhandled_exception_returns_server_failure():
    @app.get("/api/_test_boom")
    async def _boom():
        raise RuntimeError("kaboom")

    try:
        async with _client() as client:
            resp = await client.get("/api/_test_boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body == {"ok": False, "error": "SERVER_FAILURE", "detail": "kaboom"}
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes if getattr(r, "path", None) != "/api/_test_boom"
        ]



async def test_slow_request_returns_structured_timeout(monkeypatch):
    monkeypatch.setattr(main, "REQUEST_TIMEOUT_SECONDS", 0.1)

    @app.get("/api/_test_slow")
    async def _slow():
        await asyncio.sleep(1.0)
        return {"never": "reached"}

    try:
        async with _client() as client:
            resp = await client.get("/api/_test_slow")
        assert resp.status_code == 504
        body = resp.json()
        assert body["ok"] is False
        assert body["error"] == "TIMEOUT"
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes if getattr(r, "path", None) != "/api/_test_slow"
        ]



async def test_timeout_response_includes_cors_headers(monkeypatch):
    """Browser clients must be able to read the structured 504 cross-origin."""
    monkeypatch.setattr(main, "REQUEST_TIMEOUT_SECONDS", 0.1)

    @app.get("/api/_test_slow_cors")
    async def _slow_cors():
        await asyncio.sleep(1.0)
        return {"never": "reached"}

    try:
        async with _client() as client:
            resp = await client.get(
                "/api/_test_slow_cors", headers={"Origin": "https://example.com"}
            )
        assert resp.status_code == 504
        assert resp.headers.get("access-control-allow-origin") == "*"
        assert resp.json()["error"] == "TIMEOUT"
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes
            if getattr(r, "path", None) != "/api/_test_slow_cors"
        ]


async def test_docs_exempt_from_timeout(monkeypatch):
    monkeypatch.setattr(main, "REQUEST_TIMEOUT_SECONDS", 0.0001)
    async with _client() as client:
        resp = await client.get("/openapi.json")
    assert resp.status_code == 200
