"""
Tests for the retroactive OHLC malformed-candle cleanup.

Covers:
  - remove_malformed_candles() deletes bad rows and keeps good ones
  - Cleanup is idempotent (second call finds 0 rows)
  - Clean DB returns zeros in summary
  - Admin endpoint POST /admin/cleanup_malformed_candles responds correctly
    and resets the in-memory ohlc_quarantine counter
  - Concurrent cleanup + ingest upsert never deletes a valid row (race safety)
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.db import get_session
from database.models import Base, VooCandle, VixCandle, SpxCandle
from database.ohlc_cleanup import remove_malformed_candles
from main import app


# ---------------------------------------------------------------------------
# In-process async SQLite engine for tests
# ---------------------------------------------------------------------------
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine(TEST_DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s


# ---------------------------------------------------------------------------
# Helpers: build candle rows
# ---------------------------------------------------------------------------
def _good_voo(ts: datetime) -> VooCandle:
    return VooCandle(
        ticker="VOO", timestamp=ts,
        open=500.0, high=505.0, low=495.0, close=502.0,
        volume=1_000_000.0, timeframe="daily",
        is_extended_hours=False, session_type="regular",
        gap_percent=0.0, gap_type="none",
    )


def _bad_voo(ts: datetime) -> VooCandle:
    """high < open — classic yfinance glitch."""
    return VooCandle(
        ticker="VOO", timestamp=ts,
        open=680.12, high=676.71, low=674.00, close=678.00,
        volume=500_000.0, timeframe="daily",
        is_extended_hours=False, session_type="regular",
        gap_percent=0.0, gap_type="none",
    )


def _good_vix(ts: datetime) -> VixCandle:
    return VixCandle(
        ticker="^VIX", timestamp=ts,
        open=20.0, high=22.0, low=18.0, close=21.0,
        volume=0.0, timeframe="daily",
    )


def _bad_vix(ts: datetime) -> VixCandle:
    """low > close — classic yfinance glitch."""
    return VixCandle(
        ticker="^VIX", timestamp=ts,
        open=30.0, high=32.0, low=35.0, close=31.0,
        volume=0.0, timeframe="daily",
    )


def _good_spx(ts: datetime) -> SpxCandle:
    return SpxCandle(
        ticker="ES=F", timestamp=ts,
        open=5000.0, high=5050.0, low=4950.0, close=5020.0,
        volume=1_000_000.0, timeframe="daily",
    )


def _bad_spx(ts: datetime) -> SpxCandle:
    """high < open — classic yfinance glitch."""
    return SpxCandle(
        ticker="ES=F", timestamp=ts,
        open=5200.0, high=5180.0, low=5100.0, close=5190.0,
        volume=500_000.0, timeframe="daily",
    )


# ---------------------------------------------------------------------------
# Unit tests for remove_malformed_candles()
# ---------------------------------------------------------------------------

class TestRemoveMalformedCandles:

    @pytest.mark.asyncio
    async def test_deletes_bad_rows_keeps_good(self, session: AsyncSession):
        ts_good = datetime(2026, 7, 28, tzinfo=timezone.utc)
        ts_bad = datetime(2026, 7, 30, tzinfo=timezone.utc)

        session.add(_good_voo(ts_good))
        session.add(_bad_voo(ts_bad))
        await session.commit()

        summary = await remove_malformed_candles(session)
        await session.commit()

        assert summary["rows_found"] == 1
        assert summary["rows_removed"] == 1
        assert "voo_candles" in summary["tables_affected"]

        # Good row still present
        from sqlalchemy import select
        result = await session.execute(select(VooCandle))
        remaining = result.scalars().all()
        assert len(remaining) == 1
        assert remaining[0].open == 500.0

    @pytest.mark.asyncio
    async def test_idempotent_second_call_finds_zero(self, session: AsyncSession):
        ts_bad = datetime(2026, 7, 30, tzinfo=timezone.utc)
        session.add(_bad_voo(ts_bad))
        await session.commit()

        summary1 = await remove_malformed_candles(session)
        await session.commit()
        assert summary1["rows_removed"] == 1

        summary2 = await remove_malformed_candles(session)
        await session.commit()
        assert summary2["rows_found"] == 0
        assert summary2["rows_removed"] == 0
        assert summary2["tables_affected"] == []

    @pytest.mark.asyncio
    async def test_clean_db_returns_zeros(self, session: AsyncSession):
        ts = datetime(2026, 7, 28, tzinfo=timezone.utc)
        session.add(_good_voo(ts))
        await session.commit()

        summary = await remove_malformed_candles(session)
        await session.commit()

        assert summary["rows_found"] == 0
        assert summary["rows_removed"] == 0
        assert summary["tables_affected"] == []

    @pytest.mark.asyncio
    async def test_multiple_tables(self, session: AsyncSession):
        ts = datetime(2026, 7, 30, tzinfo=timezone.utc)
        session.add(_bad_voo(ts))
        session.add(_bad_vix(ts))
        await session.commit()

        summary = await remove_malformed_candles(session)
        await session.commit()

        assert summary["rows_found"] == 2
        assert summary["rows_removed"] == 2
        assert "voo_candles" in summary["tables_affected"]
        assert "vix_candles" in summary["tables_affected"]

    @pytest.mark.asyncio
    async def test_empty_db_returns_zeros(self, session: AsyncSession):
        summary = await remove_malformed_candles(session)
        await session.commit()

        assert summary["rows_found"] == 0
        assert summary["rows_removed"] == 0
        assert summary["details"]  # still has entries, just zero counts
        for d in summary["details"]:
            assert d["rows_found"] == 0
            assert d["rows_removed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_delete_is_atomic_rollback_leaves_no_rows_deleted(self, engine):
        """Bulk DELETE is atomic: a rollback after remove_malformed_candles()
        leaves every bad row intact — no partial deletion is possible.

        This verifies the single-statement DELETE WHERE id IN (...) approach:
        because the whole operation is one SQL statement inside the session's
        transaction, rolling back the transaction restores *all* rows, not just
        some of them.
        """
        factory = async_sessionmaker(engine, expire_on_commit=False)

        ts1 = datetime(2026, 5, 1, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 2, tzinfo=timezone.utc)
        ts3 = datetime(2026, 5, 3, tzinfo=timezone.utc)

        # Seed three bad rows across two tables.
        async with factory() as s:
            s.add(_bad_voo(ts1))
            s.add(_bad_voo(ts2))
            s.add(_bad_vix(ts3))
            await s.commit()

        # Run cleanup but roll back instead of committing.
        async with factory() as s:
            summary = await remove_malformed_candles(s)
            assert summary["rows_found"] == 3
            assert summary["rows_removed"] == 3
            await s.rollback()  # ← simulate crash / caller never commits

        # After rollback, ALL bad rows must still be present.
        async with factory() as s:
            voo_rows = (await s.execute(select(VooCandle))).scalars().all()
            vix_rows = (await s.execute(select(VixCandle))).scalars().all()

        assert len(voo_rows) == 2, (
            f"Expected 2 bad VOO rows to survive rollback, got {len(voo_rows)}"
        )
        assert len(vix_rows) == 1, (
            f"Expected 1 bad VIX row to survive rollback, got {len(vix_rows)}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_and_upsert_no_valid_row_deleted(self, engine):
        """Concurrent cleanup + ingest upsert must never delete a valid row.

        Simulates the startup-cleanup / incremental-ingest race: remove_malformed_candles
        and a fresh-candle upsert run as overlapping asyncio tasks on the same database.
        SQLite serialises the writes, but the two-pass logic (scan → collect bad IDs →
        delete by ID) must still leave every valid row intact regardless of ordering.
        """
        factory = async_sessionmaker(engine, expire_on_commit=False)

        # Seed: one bad row (high < open) and two valid rows.
        ts_bad = datetime(2026, 6, 1, tzinfo=timezone.utc)
        ts_good1 = datetime(2026, 6, 2, tzinfo=timezone.utc)
        ts_good2 = datetime(2026, 6, 3, tzinfo=timezone.utc)
        async with factory() as s:
            s.add(_bad_voo(ts_bad))
            s.add(_good_voo(ts_good1))
            s.add(_good_voo(ts_good2))
            await s.commit()

        # A brand-new valid candle that the "ingest" task will write concurrently.
        ts_new = datetime(2026, 6, 4, tzinfo=timezone.utc)

        async def run_cleanup() -> dict:
            async with factory() as s:
                summary = await remove_malformed_candles(s)
                await s.commit()
                return summary

        async def run_upsert() -> None:
            async with factory() as s:
                s.add(_good_voo(ts_new))
                await s.commit()

        # Launch both tasks concurrently; neither should raise.
        results = await asyncio.gather(run_cleanup(), run_upsert(), return_exceptions=True)

        for r in results:
            assert not isinstance(r, Exception), f"Concurrent task raised: {r}"

        summary = results[0]

        # Cleanup must have found and removed exactly the one malformed row.
        assert summary["rows_found"] == 1
        assert summary["rows_removed"] == 1
        assert "voo_candles" in summary["tables_affected"]

        # Verify the DB directly: only valid candles should remain.
        async with factory() as s:
            result = await s.execute(select(VooCandle))
            remaining = result.scalars().all()

        opens = {r.open for r in remaining}

        # The malformed row (open=680.12) must be gone.
        assert 680.12 not in opens, "Malformed row was not deleted"

        # Every surviving row must be a valid candle (open=500.0 in our helpers).
        for r in remaining:
            assert r.open == pytest.approx(500.0), (
                f"Valid row unexpectedly mutated or a malformed row survived: open={r.open}"
            )

        # At minimum the two pre-seeded good rows must survive; the concurrent
        # upsert row may or may not be present depending on scheduling order,
        # but if it is present it must also be valid (asserted above).
        assert len(remaining) >= 2, (
            f"Expected ≥2 valid rows after concurrent cleanup, got {len(remaining)}"
        )

    @pytest.mark.asyncio
    async def test_database_locked_on_delete_reraises_and_rollback_clean(
        self, session: AsyncSession
    ):
        """If the DB raises OperationalError during the DELETE statement,
        remove_malformed_candles() must re-raise it (not swallow it), and the
        session must still accept a rollback without a secondary error.

        This verifies the behaviour documented in the module docstring:
        "The caller is responsible for committing the session (or rolling back
        on error)."
        """
        from unittest.mock import patch
        from sqlalchemy.exc import OperationalError
        from sqlalchemy.sql.dml import Delete

        ts_bad = datetime(2026, 7, 30, tzinfo=timezone.utc)
        session.add(_bad_voo(ts_bad))
        await session.commit()

        original_execute = session.execute

        async def _patched_execute(statement, *args, **kwargs):
            # Let SELECT calls through so the scan phase works normally;
            # raise OperationalError only for the DELETE statement.
            if isinstance(statement, Delete):
                raise OperationalError(
                    "DELETE 1", {}, Exception("database is locked")
                )
            return await original_execute(statement, *args, **kwargs)

        with patch.object(session, "execute", side_effect=_patched_execute):
            with pytest.raises(OperationalError):
                await remove_malformed_candles(session)

        # Session must still accept rollback without raising a secondary error.
        await session.rollback()  # must not raise

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_cls,good_fn,bad_fn,good_open,bad_open,table_name", [
        (VixCandle, _good_vix, _bad_vix, 20.0, 30.0, "vix_candles"),
        (SpxCandle, _good_spx, _bad_spx, 5000.0, 5200.0, "spx_candles"),
    ])
    async def test_concurrent_cleanup_and_upsert_no_valid_row_deleted_vix_spx(
        self, engine, model_cls, good_fn, bad_fn, good_open, bad_open, table_name,
    ):
        """Concurrent cleanup + ingest upsert must never delete a valid VIX or SPX row.

        Mirrors the VooCandle race test but parameterised over VixCandle and SpxCandle,
        ensuring the same two-pass (scan → collect bad IDs → delete by ID) logic is
        safe for those tables too.
        """
        factory = async_sessionmaker(engine, expire_on_commit=False)

        ts_bad   = datetime(2026, 6, 1, tzinfo=timezone.utc)
        ts_good1 = datetime(2026, 6, 2, tzinfo=timezone.utc)
        ts_good2 = datetime(2026, 6, 3, tzinfo=timezone.utc)
        ts_new   = datetime(2026, 6, 4, tzinfo=timezone.utc)

        async with factory() as s:
            s.add(bad_fn(ts_bad))
            s.add(good_fn(ts_good1))
            s.add(good_fn(ts_good2))
            await s.commit()

        async def run_cleanup() -> dict:
            async with factory() as s:
                summary = await remove_malformed_candles(s)
                await s.commit()
                return summary

        async def run_upsert() -> None:
            async with factory() as s:
                s.add(good_fn(ts_new))
                await s.commit()

        results = await asyncio.gather(run_cleanup(), run_upsert(), return_exceptions=True)

        for r in results:
            assert not isinstance(r, Exception), f"Concurrent task raised: {r}"

        summary = results[0]

        # Cleanup must have found and removed exactly the one malformed row.
        assert summary["rows_found"] >= 1
        assert summary["rows_removed"] >= 1
        assert table_name in summary["tables_affected"]

        # Verify the DB directly: no malformed rows should remain.
        async with factory() as s:
            result = await s.execute(select(model_cls))
            remaining = result.scalars().all()

        opens = {r.open for r in remaining}

        # The malformed row must be gone.
        assert bad_open not in opens, (
            f"Malformed row (open={bad_open}) was not deleted from {table_name}"
        )

        # Every surviving row must be a valid candle.
        for r in remaining:
            assert r.open == pytest.approx(good_open), (
                f"Valid row unexpectedly mutated or malformed row survived in "
                f"{table_name}: open={r.open}"
            )

        # At minimum the two pre-seeded good rows must survive.
        assert len(remaining) >= 2, (
            f"Expected ≥2 valid rows in {table_name} after concurrent cleanup, "
            f"got {len(remaining)}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_cls,good_fn,bad_fn,good_open,bad_open,table_name", [
        (VixCandle, _good_vix, _bad_vix, 20.0, 30.0, "vix_candles"),
        (SpxCandle, _good_spx, _bad_spx, 5000.0, 5200.0, "spx_candles"),
    ])
    async def test_stress_concurrent_cleanup_upsert_multi_batch(
        self, engine, model_cls, good_fn, bad_fn, good_open, bad_open, table_name,
    ):
        """Stress test: >2,000 rows forces at least one full batch iteration.

        Seeds 2,100 rows (mix of good and bad) so the cleanup loop must complete
        a full BATCH=2_000 pass and then a partial second pass.  A concurrent
        upsert races against the multi-batch cleanup; every valid row must survive
        and the bad-row count must exactly match expectations.
        """
        TOTAL_ROWS = 2_100   # exceeds BATCH=2_000 → triggers second batch iteration
        BAD_EVERY = 7        # every 7th row (0-indexed) is bad → 300 bad, 1800 good

        factory = async_sessionmaker(engine, expire_on_commit=False)
        base_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)

        expected_bad = 0
        expected_good = 0

        # Insert in chunks to avoid a single enormous transaction
        CHUNK = 500
        for chunk_start in range(0, TOTAL_ROWS, CHUNK):
            async with factory() as s:
                for i in range(chunk_start, min(chunk_start + CHUNK, TOTAL_ROWS)):
                    ts = base_ts + timedelta(minutes=i)
                    if i % BAD_EVERY == 0:
                        s.add(bad_fn(ts))
                        expected_bad += 1
                    else:
                        s.add(good_fn(ts))
                        expected_good += 1
                await s.commit()

        # Concurrent upsert timestamp is safely beyond the seeded range
        ts_new = base_ts + timedelta(minutes=TOTAL_ROWS)

        async def run_cleanup() -> dict:
            async with factory() as s:
                summary = await remove_malformed_candles(s)
                await s.commit()
                return summary

        async def run_upsert() -> None:
            async with factory() as s:
                s.add(good_fn(ts_new))
                await s.commit()

        results = await asyncio.gather(run_cleanup(), run_upsert(), return_exceptions=True)

        for r in results:
            assert not isinstance(r, Exception), f"Concurrent task raised: {r}"

        summary = results[0]

        # Cleanup must have found and removed exactly the expected bad rows
        assert summary["rows_found"] == expected_bad, (
            f"{table_name}: expected {expected_bad} bad rows found, "
            f"got {summary['rows_found']}"
        )
        assert summary["rows_removed"] == expected_bad, (
            f"{table_name}: expected {expected_bad} rows removed, "
            f"got {summary['rows_removed']}"
        )
        assert table_name in summary["tables_affected"]

        # Verify the DB directly: no malformed rows should remain
        async with factory() as s:
            result = await s.execute(select(model_cls))
            remaining = result.scalars().all()

        opens = {r.open for r in remaining}

        assert bad_open not in opens, (
            f"Malformed row (open={bad_open}) survived in {table_name}"
        )

        for r in remaining:
            assert r.open == pytest.approx(good_open), (
                f"Unexpected open={r.open} in {table_name}; "
                f"malformed row may have survived or valid row mutated"
            )

        # All pre-seeded good rows must survive; the concurrent upsert row is a bonus
        assert len(remaining) >= expected_good, (
            f"Expected ≥{expected_good} valid rows in {table_name} after "
            f"multi-batch concurrent cleanup, got {len(remaining)}"
        )


# ---------------------------------------------------------------------------
# Integration test for the admin endpoint
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def http_client(engine):
    """AsyncClient wired to the FastAPI app with an isolated DB."""
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_session():
        async with factory() as s:
            yield s

    app.dependency_overrides[get_session] = _override_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.pop(get_session, None)


class TestCleanupEndpoint:

    @pytest.mark.asyncio
    async def test_endpoint_requires_admin_token(self, http_client: AsyncClient):
        resp = await http_client.post("/api/admin/cleanup_malformed_candles")
        assert resp.status_code in (403, 503)

    @pytest.mark.asyncio
    async def test_endpoint_returns_summary(self, http_client: AsyncClient, engine):
        from config import settings

        # Seed a bad candle directly via the engine
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as s:
            ts_bad = datetime(2026, 7, 30, tzinfo=timezone.utc)
            s.add(_bad_voo(ts_bad))
            await s.commit()

        token = settings.ADMIN_TOKEN or settings.SESSION_SECRET
        if not token:
            pytest.skip("ADMIN_TOKEN / SESSION_SECRET not set")

        resp = await http_client.post(
            "/api/admin/cleanup_malformed_candles",
            headers={"X-Admin-Token": token},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["rows_found"] >= 1
        assert body["rows_removed"] >= 1
        assert "voo_candles" in body["tables_affected"]
        assert "cleanup_at" in body

    @pytest.mark.asyncio
    async def test_endpoint_db_locked_returns_500(
        self, http_client: AsyncClient
    ):
        """If the DB raises OperationalError during the DELETE phase, the endpoint
        must return HTTP 500 with a human-readable error body, and the session
        teardown must complete without raising a secondary exception.

        The patch targets remove_malformed_candles (called inside the endpoint)
        to simulate the OperationalError that would arise when SQLite (or any DB)
        raises 'database is locked' mid-delete.  The session is never committed,
        so the get_session dependency's finally/rollback path runs on teardown.
        """
        from config import settings
        from unittest.mock import patch, AsyncMock
        from sqlalchemy.exc import OperationalError

        token = settings.ADMIN_TOKEN or settings.SESSION_SECRET
        if not token:
            pytest.skip("ADMIN_TOKEN / SESSION_SECRET not set")

        with patch(
            "database.ohlc_cleanup.remove_malformed_candles",
            new=AsyncMock(
                side_effect=OperationalError(
                    "DELETE FROM voo_candles WHERE id IN (?)",
                    {},
                    Exception("database is locked"),
                )
            ),
        ):
            # The ASGITransport runs the full request/response cycle including
            # dependency teardown, so any secondary exception in get_session's
            # finally block would surface here.
            resp = await http_client.post(
                "/api/admin/cleanup_malformed_candles",
                headers={"X-Admin-Token": token},
            )

        assert resp.status_code == 500, (
            f"Expected 500 on DB lock, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        # Response body must contain a human-readable error description.
        assert "detail" in body, f"No 'detail' key in error body: {body}"
        detail = body["detail"]
        assert detail, "detail field must be a non-empty string"
        # The message should reference the failure so operators know what happened.
        assert "Cleanup failed" in detail or "locked" in detail or "OperationalError" in detail, (
            f"detail does not describe the DB error: {detail!r}"
        )

    @pytest.mark.asyncio
    async def test_endpoint_commit_failure_returns_500_and_keeps_counter(
        self, http_client: AsyncClient, engine
    ):
        """session.commit() raising OperationalError must produce HTTP 500 and
        must NOT zero the quarantine counter.

        Scenario: remove_malformed_candles() completes successfully (bad rows
        are staged for deletion) but session.commit() then raises OperationalError
        (e.g. DB lock occurs at flush time).  The endpoint's except block must
        catch it and return 500 with a human-readable detail.  Because the
        counter reset lives *after* the try/commit block, the quarantine counter
        must remain at whatever value it had before the request.
        """
        from config import settings
        from unittest.mock import AsyncMock
        from sqlalchemy.exc import OperationalError
        from sqlalchemy.ext.asyncio import async_sessionmaker
        import routers.predictions as pred_mod

        token = settings.ADMIN_TOKEN or settings.SESSION_SECRET
        if not token:
            pytest.skip("ADMIN_TOKEN / SESSION_SECRET not set")

        # Pre-populate the quarantine counter so we can confirm it stays unchanged.
        pred_mod._ohlc_quarantine_stats["count"] = 7
        pred_mod._ohlc_quarantine_stats["last_ts"] = "2026-07-30T00:00:00"
        pred_mod._ohlc_quarantine_stats["last_reason"] = "high_below_open"

        factory = async_sessionmaker(engine, expire_on_commit=False)

        # Override get_session to inject a session whose commit always raises.
        async def _override_session_failing_commit():
            async with factory() as s:
                async def _failing_commit():
                    raise OperationalError(
                        "COMMIT", {}, Exception("database is locked")
                    )
                s.commit = _failing_commit
                yield s

        app.dependency_overrides[get_session] = _override_session_failing_commit
        try:
            resp = await http_client.post(
                "/api/admin/cleanup_malformed_candles",
                headers={"X-Admin-Token": token},
            )
        finally:
            # Restore the fixture's original override so subsequent tests are clean.
            async def _restore():
                async with factory() as s:
                    yield s
            app.dependency_overrides[get_session] = _restore

        assert resp.status_code == 500, (
            f"Expected 500 on commit failure, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert "detail" in body, f"No 'detail' key in error body: {body}"
        detail = body["detail"]
        assert detail, "detail field must be a non-empty string"
        assert "Cleanup failed" in detail or "locked" in detail or "OperationalError" in detail, (
            f"detail does not describe the commit error: {detail!r}"
        )

        # The quarantine counter must NOT be zeroed — the reset only runs on success.
        assert pred_mod._ohlc_quarantine_stats["count"] == 7, (
            f"Quarantine counter was incorrectly zeroed on commit failure; "
            f"got {pred_mod._ohlc_quarantine_stats['count']}"
        )

    @pytest.mark.asyncio
    async def test_endpoint_resets_quarantine_counter(
        self, http_client: AsyncClient
    ):
        """After cleanup, the in-memory ohlc_quarantine counter must be zero."""
        from config import settings
        import routers.predictions as pred_mod

        token = settings.ADMIN_TOKEN or settings.SESSION_SECRET
        if not token:
            pytest.skip("ADMIN_TOKEN / SESSION_SECRET not set")

        # Artificially inflate the counter
        pred_mod._ohlc_quarantine_stats["count"] = 5
        pred_mod._ohlc_quarantine_stats["last_ts"] = "2026-07-30T00:00:00"
        pred_mod._ohlc_quarantine_stats["last_reason"] = "high_below_open"

        resp = await http_client.post(
            "/api/admin/cleanup_malformed_candles",
            headers={"X-Admin-Token": token},
        )
        assert resp.status_code == 200

        # Counter must be zeroed
        assert pred_mod._ohlc_quarantine_stats["count"] == 0
        assert pred_mod._ohlc_quarantine_stats["last_ts"] is None
        assert pred_mod._ohlc_quarantine_stats["last_reason"] is None
