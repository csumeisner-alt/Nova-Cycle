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


# ---------------------------------------------------------------------------
# Load / integration tests: real database, no mocked sleeps
# ---------------------------------------------------------------------------
#
# Task: confirm the CLEANUP_TIMEOUT_SECONDS guard is calibrated against a
# realistic row count and that asyncio.wait_for can actually cancel the real
# remove_malformed_candles() coroutine (it only works if the coroutine yields
# to the event loop regularly — which it does, once per 2 000-row batch).
#
# Measured baseline (Replit container, July 2026): 150 000 rows scan+delete in
# ~3 s, i.e. ~50 000 rows/s.  The 300 s default budget therefore covers
# roughly 15 million rows — two orders of magnitude above the seeded volume.

import random
import time

from sqlalchemy import func, insert, select


def _seed_rows(n_rows: int, bad_every: int) -> list[dict]:
    """Build n_rows candle dicts; every `bad_every`-th row violates OHLC rules."""
    rows = []
    for i in range(n_rows):
        bad = i % bad_every == 0
        rows.append(dict(
            ticker="VOO",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc).replace(tzinfo=None),
            open=100.0,
            high=99.0 if bad else 102.0,   # bad: high < open (and < close)
            low=98.0,
            close=101.0,
            volume=1_000_000.0,
            timeframe="daily",
            is_extended_hours=False,
            session_type="regular",
        ))
    return rows


class TestCleanupLoad:
    """Integration tests against a real (in-memory) SQLite DB with >=100k rows."""

    N_ROWS = 120_000
    BAD_EVERY = 50  # 2 400 malformed rows

    @pytest.mark.asyncio
    async def test_timeout_budget_realistic_for_100k_rows(self, engine):
        """Seed >=100k rows and verify the real cleanup finishes far inside
        the 300 s budget (require at least 10x headroom)."""
        factory = async_sessionmaker(engine, expire_on_commit=False)
        rows = _seed_rows(self.N_ROWS, self.BAD_EVERY)
        async with factory() as s:
            for j in range(0, len(rows), 10_000):
                await s.execute(insert(VooCandle), rows[j:j + 10_000])
            await s.commit()

        async with factory() as s:
            t0 = time.perf_counter()
            summary = await remove_malformed_candles(s)
            await s.commit()
            elapsed = time.perf_counter() - t0

        expected_bad = self.N_ROWS // self.BAD_EVERY
        assert summary["rows_found"] == expected_bad
        assert summary["rows_removed"] == expected_bad

        # Validate CLEANUP_TIMEOUT_SECONDS is realistic: the default budget
        # must leave at least 10x headroom at this row count.
        assert _CLEANUP_TIMEOUT_SECONDS >= 300
        assert elapsed < _CLEANUP_TIMEOUT_SECONDS / 10, (
            f"cleanup of {self.N_ROWS} rows took {elapsed:.1f}s — "
            f"too close to the {_CLEANUP_TIMEOUT_SECONDS:.0f}s budget; "
            "refactor to a server-side SQL filter."
        )

        # Sanity: bad rows really gone from the DB.
        async with factory() as s:
            remaining = (await s.execute(
                select(func.count()).select_from(VooCandle)
            )).scalar_one()
        assert remaining == self.N_ROWS - expected_bad

    @pytest.mark.asyncio
    async def test_wait_for_cancels_real_cleanup_on_large_db(self, tmp_path):
        """The timeout must fire against the REAL coroutine (not a mocked
        sleep): remove_malformed_candles yields to the event loop on every
        batched SELECT, so asyncio.wait_for can cancel it mid-scan.

        Uses a file-based SQLite DB (not :memory:) because cancelling a
        query mid-flight invalidates the pooled connection — and an
        in-memory DB lives and dies with that connection.
        """
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'cleanup_load.db'}", echo=False
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        rows = _seed_rows(self.N_ROWS, self.BAD_EVERY)
        async with factory() as s:
            for j in range(0, len(rows), 10_000):
                await s.execute(insert(VooCandle), rows[j:j + 10_000])
            await s.commit()

        # A timeout far below the measured full-scan duration for this row
        # count: the scan takes ~2-3 s, so 0.2 s must interrupt it mid-flight.
        async with factory() as s:
            t0 = time.perf_counter()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(remove_malformed_candles(s), timeout=0.2)
            elapsed = time.perf_counter() - t0
            await s.rollback()
        # It must have been cut off promptly, not run to completion.
        assert elapsed < 1.5

        # Nothing was deleted (rolled back / never committed).
        async with factory() as s:
            remaining = (await s.execute(
                select(func.count()).select_from(VooCandle)
            )).scalar_one()
        assert remaining == self.N_ROWS
        await engine.dispose()
