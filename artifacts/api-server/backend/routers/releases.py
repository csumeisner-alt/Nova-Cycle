"""Latest-APK-release proxy.

The web dashboard needs the newest published APK release from GitHub. Browsers
hitting the GitHub API anonymously share a low per-IP rate limit (60/hr), so
under load the dashboard would intermittently lose its primary download CTA.

This router proxies the lookup server-side with:
- an in-memory cache (fresh for CACHE_TTL_SECONDS),
- ETag conditional requests so refreshes that hit GitHub usually cost a
  304 (which does not count against the rate limit the same way), and
- stale-serve: if GitHub is unreachable or rate-limited, the last known
  release is returned instead of failing.

Two endpoints are exposed:
- ``/latest_apk_release`` — slimmed payload (tag, published_at, apk_url) with
  ``cached``/``fetched_at`` metadata.
- ``/releases/latest`` — GitHub-shaped release object with an ``ok``/``stale``
  envelope.
"""

import asyncio
import logging
import time
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

GITHUB_RELEASES_URL = (
    "https://api.github.com/repos/csumeisner-alt/Nova-Cycle/releases?per_page=30"
)
RELEASES_API_URL = GITHUB_RELEASES_URL
CACHE_TTL_SECONDS = 300  # serve from memory without contacting GitHub

# Module-level cache state (single-process app; guarded by the event loop).
_cache: dict[str, Any] = {
    "release": None,        # selected release payload
    "etag": None,           # ETag of the last successful GitHub response
    "fetched_at": 0.0,      # monotonic time of last successful refresh
    "fetched_at_iso": None,
    "has_data": False,      # whether we've ever fetched successfully
}

# Single-flight guard: only one GitHub fetch at a time per process. Concurrent
# requests at a TTL boundary await the same refresh instead of stampeding.
_refresh_lock = asyncio.Lock()
_cache_lock = _refresh_lock


def _select_release(releases: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pick the newest versioned release that ships an APK asset.

    Skips the rolling ``latest`` alias, drafts and prereleases so the returned
    asset URL is immutable (cached redirects can never go stale).
    """
    candidates = []
    for r in releases:
        if r.get("tag_name") == "latest" or r.get("draft") or r.get("prerelease"):
            continue
        apk_assets = [
            a for a in r.get("assets", [])
            if str(a.get("name", "")).lower().endswith(".apk")
        ]
        if not apk_assets:
            continue
        # Prefer the canonical CI asset name when several APKs exist.
        apk = next(
            (a for a in apk_assets if a.get("name") == "app-release.apk"),
            apk_assets[0],
        )
        candidates.append((r.get("published_at") or "", r, apk))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, release, apk = candidates[0]
    return {
        "tag_name": release.get("tag_name"),
        "published_at": release.get("published_at"),
        "apk_url": apk.get("browser_download_url"),
        "apk_name": apk.get("name"),
        "apk_size": apk.get("size"),
    }


async def _refresh_from_github() -> None:
    headers = {"Accept": "application/vnd.github+json"}
    if _cache["etag"]:
        headers["If-None-Match"] = _cache["etag"]
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(GITHUB_RELEASES_URL, headers=headers)
    if resp.status_code == 304:
        _cache["fetched_at"] = time.monotonic()
        return
    resp.raise_for_status()
    release = _select_release(resp.json())
    _cache["release"] = release
    _cache["etag"] = resp.headers.get("etag")
    _cache["fetched_at"] = time.monotonic()
    from datetime import datetime, timezone
    _cache["fetched_at_iso"] = datetime.now(timezone.utc).isoformat()


@router.get("/latest_apk_release")
async def latest_apk_release() -> dict[str, Any]:
    """Return the newest published APK release, cached server-side."""
    def _is_stale() -> bool:
        age = time.monotonic() - _cache["fetched_at"]
        return _cache["release"] is None or age >= CACHE_TTL_SECONDS

    if _is_stale():
        async with _refresh_lock:
            # Another request may have refreshed while we waited on the lock.
            if _is_stale():
                try:
                    await _refresh_from_github()
                except Exception as exc:  # rate limit, network, 5xx — serve stale
                    logger.warning("GitHub release refresh failed: %s", exc)
                    if _cache["release"] is None:
                        raise HTTPException(
                            status_code=503,
                            detail="Unable to fetch release info from GitHub",
                        ) from exc
                    return {
                        "release": _cache["release"],
                        "cached": True,
                        "fetched_at": _cache["fetched_at_iso"],
                    }
    if _cache["release"] is None:
        raise HTTPException(status_code=404, detail="No APK release found")
    return {
        "release": _cache["release"],
        "cached": False,
        "fetched_at": _cache["fetched_at_iso"],
    }


# ---------------------------------------------------------------------------
# GitHub-shaped variant: /releases/latest
# Keeps its own cache slot ("gh_release") because it returns the full release
# object (assets list) rather than the slimmed payload above.
# ---------------------------------------------------------------------------


def _select_latest_release(releases: list[dict]) -> Optional[dict]:
    """Pick the newest versioned (non-draft, non-prerelease) release with an APK.

    Mirrors the dashboard's original client-side selection: skip the rolling
    'latest' alias so the link targets an immutable asset URL.
    """
    versioned = [
        r
        for r in releases
        if r.get("tag_name") != "latest"
        and not r.get("draft")
        and not r.get("prerelease")
        and any(
            (a.get("name") or "").lower().endswith(".apk")
            for a in r.get("assets", [])
        )
    ]
    versioned.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    if not versioned:
        return None
    r = versioned[0]
    return {
        "tag_name": r.get("tag_name"),
        "name": r.get("name"),
        "published_at": r.get("published_at"),
        "draft": bool(r.get("draft")),
        "prerelease": bool(r.get("prerelease")),
        "assets": [
            {
                "name": a.get("name"),
                "browser_download_url": a.get("browser_download_url"),
                "size": a.get("size"),
            }
            for a in r.get("assets", [])
        ],
    }


async def _refresh_cache() -> None:
    """Fetch (or revalidate) full release metadata from GitHub for
    /releases/latest, updating the cache.

    Raises on failure; callers decide whether stale data can be served.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "novacycle-api",
    }
    if _cache.get("gh_etag"):
        headers["If-None-Match"] = _cache["gh_etag"]

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(RELEASES_API_URL, headers=headers)

    if resp.status_code == 304:
        # Unchanged — refresh the TTL, keep the cached payload.
        _cache["gh_fetched_at"] = time.monotonic()
        return

    resp.raise_for_status()
    releases = resp.json()
    _cache["gh_release"] = _select_latest_release(releases)
    _cache["gh_etag"] = resp.headers.get("etag")
    _cache["gh_fetched_at"] = time.monotonic()
    _cache["has_data"] = True


@router.get("/releases/latest")
async def latest_release():
    """Latest APK release metadata, cached and resilient to GitHub outages."""
    async with _cache_lock:
        fresh = (
            _cache.get("has_data")
            and time.monotonic() - _cache.get("gh_fetched_at", 0.0)
            < CACHE_TTL_SECONDS
        )
        if not fresh:
            try:
                await _refresh_cache()
            except Exception as e:
                if _cache.get("has_data"):
                    logger.warning(
                        f"GitHub release fetch failed; serving cached data: {e}"
                    )
                    return JSONResponse(
                        {
                            "ok": True,
                            "release": _cache.get("gh_release"),
                            "stale": True,
                        }
                    )
                logger.error(f"GitHub release fetch failed with no cache: {e}")
                return JSONResponse(
                    status_code=502,
                    content={
                        "ok": False,
                        "error": "UPSTREAM_UNAVAILABLE",
                        "detail": "Could not fetch release info from GitHub",
                    },
                )
        return JSONResponse(
            {"ok": True, "release": _cache.get("gh_release"), "stale": False}
        )
