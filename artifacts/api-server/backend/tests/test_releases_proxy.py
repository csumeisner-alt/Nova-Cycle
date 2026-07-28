"""Tests for the GitHub release proxy (/api/releases/latest)."""

import time

import pytest

from routers import releases


def _rel(tag, published, draft=False, prerelease=False, assets=None):
    return {
        "tag_name": tag,
        "name": f"Release {tag}",
        "published_at": published,
        "draft": draft,
        "prerelease": prerelease,
        "assets": assets
        if assets is not None
        else [
            {
                "name": "app-release.apk",
                "browser_download_url": f"https://example.com/{tag}/app-release.apk",
                "size": 123,
            }
        ],
    }


class TestSelectLatestRelease:
    def test_picks_newest_versioned_apk_release(self):
        out = releases._select_latest_release(
            [
                _rel("v1", "2026-07-01T00:00:00Z"),
                _rel("v2", "2026-07-02T00:00:00Z"),
            ]
        )
        assert out["tag_name"] == "v2"

    def test_skips_latest_alias_drafts_and_prereleases(self):
        out = releases._select_latest_release(
            [
                _rel("latest", "2026-07-09T00:00:00Z"),
                _rel("v3-draft", "2026-07-08T00:00:00Z", draft=True),
                _rel("v3-rc", "2026-07-07T00:00:00Z", prerelease=True),
                _rel("v2", "2026-07-02T00:00:00Z"),
            ]
        )
        assert out["tag_name"] == "v2"

    def test_skips_releases_without_apk(self):
        out = releases._select_latest_release(
            [
                _rel("v2", "2026-07-02T00:00:00Z", assets=[{"name": "notes.txt"}]),
                _rel("v1", "2026-07-01T00:00:00Z"),
            ]
        )
        assert out["tag_name"] == "v1"

    def test_returns_none_when_no_candidates(self):
        assert releases._select_latest_release([]) is None


@pytest.mark.asyncio
class TestLatestReleaseEndpoint:
    def setup_method(self):
        releases._cache.update(
            {"gh_release": None, "gh_etag": None, "gh_fetched_at": 0.0, "has_data": False}
        )

    async def test_serves_cached_when_fresh(self, monkeypatch):
        releases._cache.update(
            {
                "gh_release": {"tag_name": "v9"},
                "has_data": True,
                "gh_fetched_at": time.monotonic(),
            }
        )

        async def boom():
            raise AssertionError("should not refetch while fresh")

        monkeypatch.setattr(releases, "_refresh_cache", boom)
        resp = await releases.latest_release()
        assert resp.status_code == 200
        assert b'"stale":false' in resp.body.replace(b" ", b"")

    async def test_serves_stale_cache_when_github_fails(self, monkeypatch):
        releases._cache.update(
            {
                "gh_release": {"tag_name": "v9"},
                "has_data": True,
                "gh_fetched_at": time.monotonic() - releases.CACHE_TTL_SECONDS - 1,
            }
        )

        async def boom():
            raise RuntimeError("rate limited")

        monkeypatch.setattr(releases, "_refresh_cache", boom)
        resp = await releases.latest_release()
        assert resp.status_code == 200
        body = resp.body.replace(b" ", b"")
        assert b'"stale":true' in body
        assert b'"v9"' in body

    async def test_errors_structured_when_no_cache(self, monkeypatch):
        async def boom():
            raise RuntimeError("rate limited")

        monkeypatch.setattr(releases, "_refresh_cache", boom)
        resp = await releases.latest_release()
        assert resp.status_code == 502
        assert b"UPSTREAM_UNAVAILABLE" in resp.body
