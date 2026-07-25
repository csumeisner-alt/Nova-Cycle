"""Tests: the 5-min feed recovery record survives a simulated backend restart."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VooCandle
from ingestion import market_calendar, pipeline
from ingestion.pipeline import (
    IngestionPipeline,
    check_5min_staleness,
    get_5min_recovery_status,
)
from ingestion.recovery_history import (
    get_persisted_recovery_status,
    record_recovery_attempt,
)


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _bar(ts):
    return VooCandle(
        ticker=settings.TICKER, timestamp=ts, open=1, high=1, low=1, close=1,
        volume=1, timeframe="5min", is_extended_hours=False,
        session_type="regular", gap_percent=0.0, gap_type="none",
    )


def _midday_trading_utc() -> datetime:
    d = datetime.utcnow().date()
    while True:
        if market_calendar.is_trading_day(d) and not market_calendar.is_half_day(d):
            et_noon = datetime(d.year, d.month, d.day, 12, 0, tzinfo=market_calendar.EASTERN)
            return et_noon.astimezone(market_calendar.timezone.utc).replace(tzinfo=None)
        d -= timedelta(days=1)


NOW = _midday_trading_utc()
MAX_AGE = settings.FIVEMIN_STALENESS_MAX_AGE_MINUTES


def _simulate_restart():
    """Wipe the in-memory record, as a process restart would."""
    pipeline._last_5min_recovery_status.update(
        {"last_attempt_at": None, "outcome": None, "bars_fetched": None}
    )


def test_record_and_load_roundtrip():
    record_recovery_attempt("failed", "2026-07-25T14:00:00", 0)
    record_recovery_attempt("recovered", "2026-07-25T14:20:00", 7)

    data = get_persisted_recovery_status()
    assert data["last_attempt"] == {
        "last_attempt_at": "2026-07-25T14:20:00",
        "outcome": "recovered",
        "bars_fetched": 7,
    }
    assert len(data["history"]) == 2
    assert data["failure_count"] == 1


def test_history_is_capped():
    from ingestion import recovery_history

    for i in range(recovery_history.MAX_HISTORY + 5):
        record_recovery_attempt("failed", f"2026-07-25T14:{i:02d}:00", 0)
    data = get_persisted_recovery_status()
    assert len(data["history"]) == recovery_history.MAX_HISTORY
    # Oldest entries were trimmed; failure count still cumulative.
    assert data["failure_count"] == recovery_history.MAX_HISTORY + 5
    assert data["history"][-1]["last_attempt_at"].endswith(
        f"{recovery_history.MAX_HISTORY + 4}:00"
    )


def test_corrupt_file_never_raises(tmp_path, monkeypatch):
    from ingestion import recovery_history

    bad = tmp_path / "corrupt.json"
    bad.write_text("{not json")
    monkeypatch.setattr(recovery_history, "HISTORY_PATH", bad)

    assert get_persisted_recovery_status() == {
        "last_attempt": None, "history": [], "failure_count": 0,
    }
    record_recovery_attempt("failed", "2026-07-25T14:00:00", 0)  # must not raise
    assert get_persisted_recovery_status()["failure_count"] == 1


@pytest.mark.asyncio
async def test_status_survives_simulated_restart(db_session):
    """A failed recovery attempt remains visible after the process restarts."""
    db_session.add(_bar(NOW - timedelta(minutes=MAX_AGE + 30)))
    await db_session.flush()

    p = IngestionPipeline()
    p.fetcher.fetch_5min_range = AsyncMock(return_value=pd.DataFrame())
    status = await check_5min_staleness(db_session, now=NOW)
    await p.recover_5min_stall(status, db_session, now=NOW)

    before = get_5min_recovery_status()
    assert before["outcome"] == "failed"
    assert before["from_previous_run"] is False

    _simulate_restart()

    after = get_5min_recovery_status()
    assert after["outcome"] == "failed"
    assert after["last_attempt_at"] == NOW.isoformat()
    assert after["bars_fetched"] == 0
    assert after["from_previous_run"] is True
    assert after["failure_count"] == 1
    assert len(after["history"]) == 1


@pytest.mark.asyncio
async def test_inmemory_record_wins_over_persisted(db_session):
    """A fresh in-process attempt overrides the persisted (older) record."""
    record_recovery_attempt("failed", "2020-01-01T00:00:00", 0)

    db_session.add(_bar(NOW - timedelta(minutes=MAX_AGE + 30)))
    await db_session.flush()

    p = IngestionPipeline()
    p.fetcher.fetch_5min_range = AsyncMock(return_value=pd.DataFrame())
    status = await check_5min_staleness(db_session, now=NOW)
    await p.recover_5min_stall(status, db_session, now=NOW)

    rec = get_5min_recovery_status()
    assert rec["last_attempt_at"] == NOW.isoformat()
    assert rec["from_previous_run"] is False
    # Both attempts are in the persisted history / failure count.
    assert rec["failure_count"] == 2
    assert len(rec["history"]) == 2


@pytest.mark.asyncio
async def test_healthz_serves_persisted_record_after_restart(db_session):
    """/healthz shows the persisted record (and failure alert) post-restart."""
    from routers.predictions import healthz

    db_session.add(_bar(NOW - timedelta(minutes=MAX_AGE + 30)))
    await db_session.flush()

    p = IngestionPipeline()
    p.fetcher.fetch_5min_range = AsyncMock(return_value=pd.DataFrame())
    status = await check_5min_staleness(db_session, now=NOW)
    await p.recover_5min_stall(status, db_session, now=NOW)

    _simulate_restart()

    body = await healthz(session=db_session)
    rec = body["voo_5min_recovery"]
    assert rec["outcome"] == "failed"
    assert rec["from_previous_run"] is True
    assert any(a.startswith("voo_5min_recovery:") for a in body["alerts"])
