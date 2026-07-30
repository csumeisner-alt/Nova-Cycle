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
from datetime import datetime, timezone

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
    """low > close."""
    return VixCandle(
        ticker="^VIX", timestamp=ts,
        open=20.0, high=22.0, low=25.0, close=21.0,
        volume=0.0, timeframe="daily",
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
