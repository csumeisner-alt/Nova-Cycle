"""
Tests for the background OHLC startup-cleanup task.

Covers:
  - cleanup_pending is True while the task is running, False when it finishes
  - cleanup_pending is False after a successful run
  - The 300 s timeout guard fires and cleans up state (cleanup_pending → False)
  - /api/healthz includes a cleanup_pending field
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.cleanup_state import (
    is_cleanup_done,
    is_cleanup_pending,
    mark_cleanup_finished,
    mark_cleanup_started,
    reset_for_testing,
)
from database.db import get_session
from database.models import Base, VooCandle
from database.ohlc_cleanup import remove_malformed_candles
from main import app, _run_startup_cleanup, _CLEANUP_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(autouse=True)
async def reset_state():
    """Reset cleanup_state module globals before each test."""
    reset_for_testing()
    yield
    reset_for_testing()


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine(TEST_DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


# ---------------------------------------------------------------------------
# cleanup_state unit tests
# ---------------------------------------------------------------------------

class TestCleanupState:

    def test_initial_state(self):
        assert is_cleanup_pending() is False
        assert is_cleanup_done() is False

    def test_mark_started(self):
        mark_cleanup_started()
        assert is_cleanup_pending() is True
        assert is_cleanup_done() is False

    def test_mark_finished(self):
        mark_cleanup_started()
        mark_cleanup_finished()
        assert is_cleanup_pending() is False
        assert is_cleanup_done() is True

    def test_reset_for_testing(self):
        mark_cleanup_started()
        mark_cleanup_finished()
        reset_for_testing()
        assert is_cleanup_pending() is False
        assert is_cleanup_done() is False


# ---------------------------------------------------------------------------
# Background task behaviour
# ---------------------------------------------------------------------------

class TestRunStartupCleanup:

    @pytest.mark.asyncio
    async def test_pending_true_while_running_then_false_on_finish(self):
        """cleanup_pending is True during execution and False afterwards."""
        barrier = asyncio.Event()
        released = asyncio.Event()

        async def _slow_cleanup(session):
            # Signal that we're inside the cleanup, then wait.
            released.set()
            await barrier.wait()
            return {
                "rows_found": 0, "rows_removed": 0,
                "tables_affected": [], "timeframes_affected": [], "details": [],
            }

        with patch("main.remove_malformed_candles", side_effect=_slow_cleanup):
            task = asyncio.create_task(_run_startup_cleanup())
            # Wait until we are inside the slow cleanup coroutine.
            await asyncio.wait_for(released.wait(), timeout=5)
            assert is_cleanup_pending() is True

            # Release the barrier so the task can finish.
            barrier.set()
            await asyncio.wait_for(task, timeout=5)

        assert is_cleanup_pending() is False
        assert is_cleanup_done() is True

    @pytest.mark.asyncio
    async def test_pending_false_after_clean_db(self):
        """cleanup_pending is False and cleanup_done is True after a no-op run."""
        async def _instant_cleanup(session):
            return {
                "rows_found": 0, "rows_removed": 0,
                "tables_affected": [], "timeframes_affected": [], "details": [],
            }

        with patch("main.remove_malformed_candles", side_effect=_instant_cleanup):
            await _run_startup_cleanup()

        assert is_cleanup_pending() is False
        assert is_cleanup_done() is True

    @pytest.mark.asyncio
    async def test_timeout_guard_sets_pending_false(self):
        """If cleanup exceeds the timeout, cleanup_pending is still reset to False."""
        async def _hung_cleanup(session):
            await asyncio.sleep(9999)  # will be cancelled by wait_for

        with (
            patch("main.remove_malformed_candles", side_effect=_hung_cleanup),
            patch("main._CLEANUP_TIMEOUT_SECONDS", 0.05),  # 50 ms timeout
        ):
            await _run_startup_cleanup()

        assert is_cleanup_pending() is False
        assert is_cleanup_done() is True

    @pytest.mark.asyncio
    async def test_exception_in_cleanup_sets_pending_false(self):
        """If cleanup raises an unexpected error, cleanup_pending is still reset."""
        async def _failing_cleanup(session):
            raise RuntimeError("simulated database failure")

        with patch("main.remove_malformed_candles", side_effect=_failing_cleanup):
            # Must not propagate the exception.
            await _run_startup_cleanup()

        assert is_cleanup_pending() is False
        assert is_cleanup_done() is True


# ---------------------------------------------------------------------------
# Healthz integration: cleanup_pending field present
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def http_client(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_session():
        async with factory() as s:
            yield s

    app.dependency_overrides[get_session] = _override_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.pop(get_session, None)


class TestHealthzCleanupPending:

    @pytest.mark.asyncio
    async def test_healthz_includes_cleanup_pending_false_by_default(
        self, http_client: AsyncClient
    ):
        reset_for_testing()
        resp = await http_client.get("/api/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert "cleanup_pending" in body
        assert body["cleanup_pending"] is False

    @pytest.mark.asyncio
    async def test_healthz_cleanup_pending_true_while_running(
        self, http_client: AsyncClient
    ):
        mark_cleanup_started()
        resp = await http_client.get("/api/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cleanup_pending"] is True
        # Reset so other tests are not affected.
        mark_cleanup_finished()

    @pytest.mark.asyncio
    async def test_healthz_cleanup_pending_false_after_finish(
        self, http_client: AsyncClient
    ):
        mark_cleanup_started()
        mark_cleanup_finished()
        resp = await http_client.get("/api/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cleanup_pending"] is False
