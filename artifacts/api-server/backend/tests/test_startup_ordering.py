"""
Tests confirming that scheduled ingest jobs cannot run before
IngestionPipeline.initialize() completes.

The guard is an asyncio.Event (_initialized_event) set in the finally-block
of initialize(). Every scheduled job calls pipeline.wait_for_initialized()
before touching the database, ensuring cleanup and ingest never overlap on
a cold start even when the scheduler fires inside the startup window.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.models import Base
from ingestion.pipeline import IngestionPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _make_pipeline() -> IngestionPipeline:
    """Return a pipeline whose heavy I/O is fully mocked."""
    p = IngestionPipeline()
    p.fetcher = MagicMock()
    p.fetcher.fetch_historical_voo = AsyncMock(return_value={})
    p.fetcher.fetch_historical_vix = AsyncMock(return_value=MagicMock(empty=True))
    p.fetcher.fetch_historical_spx = AsyncMock(return_value=MagicMock(empty=True))
    p.fetcher.fetch_incremental_voo = AsyncMock(return_value={})
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Guard unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInitializedGuard:
    """Unit-level checks of the _initialized_event / wait_for_initialized API."""

    @pytest.mark.asyncio
    async def test_event_not_set_before_initialize(self):
        """The event must not be pre-set on a fresh pipeline instance."""
        p = _make_pipeline()
        event = p._get_initialized_event()
        assert not event.is_set(), (
            "Scheduled jobs would run without waiting if the event were set at "
            "construction time."
        )

    @pytest.mark.asyncio
    async def test_wait_for_initialized_blocks_until_set(self):
        """wait_for_initialized() must not return before the event is set."""
        p = _make_pipeline()

        released = False

        async def waiter():
            await p.wait_for_initialized()
            nonlocal released
            released = True

        # Start the waiter; it should stall.
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)          # yield so the waiter runs up to the await
        assert not released, "wait_for_initialized returned before event was set"

        # Simulate initialize() completing.
        p._initialized_flag = True
        p._get_initialized_event().set()
        await asyncio.sleep(0)          # yield so the waiter can unblock

        assert released, "wait_for_initialized did not unblock after event.set()"
        task.cancel()

    @pytest.mark.asyncio
    async def test_wait_for_initialized_returns_immediately_when_already_set(self):
        """If initialize() already completed, wait_for_initialized() is a no-op."""
        p = _make_pipeline()
        p._initialized_flag = True
        p._get_initialized_event().set()

        # Should complete without yielding.
        await asyncio.wait_for(p.wait_for_initialized(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_lazy_event_creation_respects_existing_flag(self):
        """Event created after initialize() already ran must be pre-set."""
        p = _make_pipeline()
        p._initialized_flag = True
        # Force lazy creation — no event exists yet.
        event = p._get_initialized_event()
        assert event.is_set(), (
            "Lazily created event should be set immediately when "
            "_initialized_flag is already True."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ordering integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestStartupOrdering:
    """Verify the scheduled-job / initialize() ordering guarantee end-to-end."""

    @pytest.mark.asyncio
    async def test_scheduled_job_waits_for_initialize(self, db_session):
        """
        Simulate the cold-start race:
          - initialize() is slow (artificial delay)
          - a scheduled incremental-update job fires concurrently

        The job must not start its DB work until after initialize() returns.
        """
        p = _make_pipeline()

        init_finished_at: list[float] = []
        job_started_at: list[float] = []

        # Wrap initialize() so it records when it finishes.
        original_initialize = IngestionPipeline.initialize

        async def slow_initialize(self_, session):
            await asyncio.sleep(0.05)   # simulate slow full-history fetch
            await original_initialize(self_, session)
            init_finished_at.append(asyncio.get_event_loop().time())

        # Simulate the scheduled job: it awaits wait_for_initialized() then
        # records when its "DB work" begins — mirroring _run_incremental_update
        # in main.py.
        async def simulated_scheduled_job():
            await p.wait_for_initialized()
            job_started_at.append(asyncio.get_event_loop().time())

        with patch.object(IngestionPipeline, "initialize", slow_initialize):
            # Patch away heavy sub-calls inside initialize that hit network / DB.
            with patch.object(p, "remove_invalid_voo_candles", new=AsyncMock(return_value=0)), \
                 patch.object(p, "run_incremental_update", new=AsyncMock()):

                init_task = asyncio.create_task(p.initialize(db_session))
                # Give the init task a head-start so it's firmly inside the
                # slow_initialize sleep when the job fires.
                await asyncio.sleep(0)
                job_task = asyncio.create_task(simulated_scheduled_job())

                await asyncio.gather(init_task, job_task)

        assert len(init_finished_at) == 1, "initialize() should have run exactly once"
        assert len(job_started_at) == 1, "scheduled job should have run exactly once"
        assert job_started_at[0] >= init_finished_at[0], (
            f"Scheduled job started ({job_started_at[0]:.6f}) BEFORE "
            f"initialize() finished ({init_finished_at[0]:.6f}) — "
            "cleanup and ingest overlapped."
        )

    @pytest.mark.asyncio
    async def test_initialize_sets_event_even_on_failure(self, db_session):
        """
        If initialize() raises, the event must still be set so that scheduled
        jobs are not blocked forever.
        """
        p = _make_pipeline()

        async def failing_initialize(session):
            raise RuntimeError("simulated yfinance outage")

        with patch.object(p, "remove_invalid_voo_candles",
                          new=AsyncMock(side_effect=RuntimeError("simulated"))):
            try:
                await p.initialize(db_session)
            except RuntimeError:
                pass  # expected

        assert p._initialized_flag, (
            "initialize() must set _initialized_flag even after an exception "
            "so scheduled jobs are not blocked forever."
        )
        assert p._get_initialized_event().is_set(), (
            "asyncio.Event must be set after a failed initialize() so that "
            "wait_for_initialized() can unblock."
        )

    @pytest.mark.asyncio
    async def test_scheduled_jobs_unblock_when_pre_initialize_step_crashes(self):
        """
        If _init_pipeline() in main.py crashes before pipeline.initialize() is
        ever called (e.g. reclassify_session_labels raises), the outer finally
        must still release the initialized guard so scheduled jobs don't hang.

        This test mirrors what the outer try/finally in main._init_pipeline()
        guarantees: even without going through pipeline.initialize(), calling
        pipeline._get_initialized_event().set() directly unblocks all waiters.
        """
        p = _make_pipeline()

        unblocked: list[bool] = []

        async def job():
            await p.wait_for_initialized()
            unblocked.append(True)

        job_task = asyncio.create_task(job())
        await asyncio.sleep(0)  # let the job park on the event
        assert not unblocked, "Job should be blocked before guard is released"

        # Simulate the _init_pipeline finally-block: pre-init step failed,
        # initialize() was never called, but the guard is released anyway.
        p._initialized_flag = True
        p._get_initialized_event().set()

        await job_task
        assert unblocked, "Job must unblock even when initialize() was never called"

    @pytest.mark.asyncio
    async def test_multiple_jobs_all_unblock_after_initialize(self, db_session):
        """
        Multiple concurrent scheduled jobs must all unblock once initialize()
        completes, not just the first waiter.
        """
        p = _make_pipeline()
        unblocked: list[int] = []

        async def job(job_id: int):
            await p.wait_for_initialized()
            unblocked.append(job_id)

        with patch.object(p, "remove_invalid_voo_candles", new=AsyncMock(return_value=0)), \
             patch.object(p, "run_incremental_update", new=AsyncMock()):

            job_tasks = [asyncio.create_task(job(i)) for i in range(5)]
            await asyncio.sleep(0)          # let all jobs park on the event
            assert unblocked == [], "Jobs should be blocked before initialize()"

            await p.initialize(db_session)
            await asyncio.gather(*job_tasks)

        assert sorted(unblocked) == list(range(5)), (
            "All 5 concurrent scheduled jobs should have unblocked after "
            "initialize() completed."
        )
