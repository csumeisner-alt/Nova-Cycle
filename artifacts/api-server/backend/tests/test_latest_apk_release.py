"""Tests for the /api/latest_apk_release proxy endpoint."""

import time

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from main import app
from routers import releases


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _release(tag, published_at, assets, draft=False, prerelease=False):
    return {
        "tag_name": tag,
        "name": f"Release {tag}",
        "published_at": published_at,
        "draft": draft,
        "prerelease": prerelease,
        "assets": assets,
    }


def _apk(name="app-release.apk", url="https://example.com/app-release.apk"):
    return {"name": name, "browser_download_url": url, "size": 123}


@pytest.fixture(autouse=True)
def reset_cache():
    releases._cache.update(
        {"release": None, "etag": None, "fetched_at": 0.0, "fetched_at_iso": None}
    )
    yield


class TestSelectRelease:
    def test_picks_newest_versioned_release_with_apk(self):
        picked = releases._select_release([
            _release("latest", "2026-07-28T02:00:00Z", [_apk()]),
            _release("v2026-07-28-abc1234", "2026-07-28T02:00:00Z", [_apk()]),
            _release("v2026-07-20-old0000", "2026-07-20T02:00:00Z", [_apk()]),
        ])
        assert picked["tag_name"] == "v2026-07-28-abc1234"
        assert picked["apk_url"] == "https://example.com/app-release.apk"

    def test_skips_drafts_prereleases_and_apkless_releases(self):
        picked = releases._select_release([
            _release("v3", "2026-07-30T00:00:00Z", [_apk()], draft=True),
            _release("v2", "2026-07-29T00:00:00Z", [_apk()], prerelease=True),
            _release("v1-noapk", "2026-07-28T00:00:00Z", [{"name": "notes.txt",
                     "browser_download_url": "u", "size": 1}]),
            _release("v0", "2026-07-27T00:00:00Z", [_apk()]),
        ])
        assert picked["tag_name"] == "v0"

    def test_prefers_canonical_apk_name(self):
        picked = releases._select_release([
            _release("v1", "2026-07-28T00:00:00Z", [
                _apk("other.APK", "https://example.com/other.apk"),
                _apk("app-release.apk", "https://example.com/canonical.apk"),
            ]),
        ])
        assert picked["apk_url"] == "https://example.com/canonical.apk"

    def test_returns_none_when_no_candidates(self):
        assert releases._select_release([]) is None


class TestEndpoint:
    @pytest.mark.asyncio
    async def test_serves_fresh_cache_without_refetch(self, client):
        releases._cache.update({
            "release": {"tag_name": "v1", "apk_url": "u", "published_at": None},
            "fetched_at": time.monotonic(),
            "fetched_at_iso": "2026-07-28T00:00:00+00:00",
        })
        resp = await client.get("/api/latest_apk_release")
        assert resp.status_code == 200
        body = resp.json()
        assert body["release"]["tag_name"] == "v1"
        assert body["cached"] is False

    @pytest.mark.asyncio
    async def test_serves_stale_cache_when_github_fails(self, client, monkeypatch):
        releases._cache.update({
            "release": {"tag_name": "v1", "apk_url": "u", "published_at": None},
            "fetched_at": 0.0,  # expired
            "fetched_at_iso": "2026-07-28T00:00:00+00:00",
        })

        async def boom():
            raise RuntimeError("rate limited")

        monkeypatch.setattr(releases, "_refresh_from_github", boom)
        resp = await client.get("/api/latest_apk_release")
        assert resp.status_code == 200
        body = resp.json()
        assert body["release"]["tag_name"] == "v1"
        assert body["cached"] is True

    @pytest.mark.asyncio
    async def test_503_when_no_cache_and_github_fails(self, client, monkeypatch):
        async def boom():
            raise RuntimeError("down")

        monkeypatch.setattr(releases, "_refresh_from_github", boom)
        resp = await client.get("/api/latest_apk_release")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_concurrent_requests_trigger_single_refresh(
        self, client, monkeypatch
    ):
        import asyncio

        calls = 0

        async def slow_refresh():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            releases._cache.update({
                "release": {"tag_name": "v9", "apk_url": "u", "published_at": None},
                "fetched_at": time.monotonic(),
                "fetched_at_iso": "2026-07-28T00:00:00+00:00",
            })

        monkeypatch.setattr(releases, "_refresh_from_github", slow_refresh)
        responses = await asyncio.gather(
            *[client.get("/api/latest_apk_release") for _ in range(5)]
        )
        assert all(r.status_code == 200 for r in responses)
        assert calls == 1  # single-flight: no request stampede to GitHub

    @pytest.mark.asyncio
    async def test_etag_304_refreshes_ttl_without_replacing_release(
        self, client, monkeypatch
    ):
        # Cache holds an expired release with an ETag; GitHub answers 304.
        releases._cache.update({
            "release": {"tag_name": "v1", "apk_url": "u", "published_at": None},
            "etag": 'W/"abc"',
            "fetched_at": 0.0,  # expired
            "fetched_at_iso": "2026-07-28T00:00:00+00:00",
        })

        seen_headers = {}

        class FakeResponse:
            status_code = 304
            headers = {}

        class FakeClient:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, headers=None):
                seen_headers.update(headers or {})
                return FakeResponse()

        monkeypatch.setattr(releases.httpx, "AsyncClient", FakeClient)
        resp = await client.get("/api/latest_apk_release")
        assert resp.status_code == 200
        body = resp.json()
        assert body["release"]["tag_name"] == "v1"  # kept, not replaced
        assert body["cached"] is False  # 304 revalidation counts as fresh
        assert seen_headers.get("If-None-Match") == 'W/"abc"'
        assert releases._cache["fetched_at"] > 0.0  # TTL refreshed
