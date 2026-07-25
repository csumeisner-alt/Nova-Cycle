"""Tests for the immediate targeted re-fetch when a 5-min feed stall is detected."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VooCandle
from ingestion import market_calendar
from ingestion.pipeline import IngestionPipeline, check_5min_staleness


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


def _fresh_frame(now: datetime, count: int = 4) -> pd.DataFrame:
    """A frame of fresh 5-min bars ending a few minutes before `now`."""
    idx = pd.DatetimeIndex(
        [now - timedelta(minutes=5 * (count - i)) for i in range(count)]
    )
    return pd.DataFrame(
        {
            "open": [1.0] * count, "high": [1.0] * count,
            "low": [1.0] * count, "close": [1.0] * count,
            "volume": [1.0] * count,
            "is_extended_hours": [False] * count,
            "session_type": ["regular"] * count,
        },
        index=idx,
    )


@pytest.mark.asyncio
async def test_recovery_refetches_and_clears_staleness(db_session, caplog):
    stale_ts = NOW - timedelta(minutes=MAX_AGE + 30)
    db_session.add(_bar(stale_ts))
    await db_session.flush()

    pipeline = IngestionPipeline()
    pipeline.fetcher.fetch_5min_range = AsyncMock(return_value=_fresh_frame(NOW))

    status = await check_5min_staleness(db_session, now=NOW)
    assert status["stale"] is True

    with caplog.at_level("INFO"):
        summary = await pipeline.recover_5min_stall(status, db_session, now=NOW)

    assert summary["attempted"] is True
    assert summary["recovered"] is True
    assert summary["bars_fetched"] > 0
    pipeline.fetcher.fetch_5min_range.assert_awaited_once()
    # Fetch window starts at the last stored bar.
    call_start = pipeline.fetcher.fetch_5min_range.await_args.args[0]
    assert call_start == stale_ts
    assert any("fivemin_stall_recovered" in r.message for r in caplog.records)

    recheck = await check_5min_staleness(db_session, now=NOW)
    assert recheck["stale"] is False


@pytest.mark.asyncio
async def test_recovery_failure_logged_distinctly(db_session, caplog):
    db_session.add(_bar(NOW - timedelta(minutes=MAX_AGE + 30)))
    await db_session.flush()

    pipeline = IngestionPipeline()
    pipeline.fetcher.fetch_5min_range = AsyncMock(return_value=pd.DataFrame())

    status = await check_5min_staleness(db_session, now=NOW)
    with caplog.at_level("INFO"):
        summary = await pipeline.recover_5min_stall(status, db_session, now=NOW)

    assert summary["attempted"] is True
    assert summary["recovered"] is False
    assert summary["bars_fetched"] == 0
    assert any("fivemin_stall_recovery_failed" in r.message for r in caplog.records)
    assert not any("fivemin_stall_recovered " in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_recovery_with_no_bars_stored(db_session):
    """When no 5-min bars exist at all, recovery fetches from `now`'s day."""
    pipeline = IngestionPipeline()
    pipeline.fetcher.fetch_5min_range = AsyncMock(return_value=_fresh_frame(NOW))

    status = await check_5min_staleness(db_session, now=NOW)
    assert status["stale"] is True and status["latest_5min"] is None

    summary = await pipeline.recover_5min_stall(status, db_session, now=NOW)
    assert summary["attempted"] is True
    assert summary["recovered"] is True


@pytest.mark.asyncio
async def test_cooldown_prevents_tight_retry_loop(db_session, caplog):
    db_session.add(_bar(NOW - timedelta(minutes=MAX_AGE + 30)))
    await db_session.flush()

    pipeline = IngestionPipeline()
    pipeline.fetcher.fetch_5min_range = AsyncMock(return_value=pd.DataFrame())

    status = await check_5min_staleness(db_session, now=NOW)
    first = await pipeline.recover_5min_stall(status, db_session, now=NOW)
    assert first["attempted"] is True

    # Second attempt 5 minutes later (next scheduler tick) is skipped.
    with caplog.at_level("INFO"):
        second = await pipeline.recover_5min_stall(
            status, db_session, now=NOW + timedelta(minutes=5)
        )
    assert second["attempted"] is False
    assert second["reason"] == "cooldown"
    assert pipeline.fetcher.fetch_5min_range.await_count == 1
    assert any("fivemin_stall_recovery_skipped" in r.message for r in caplog.records)

    # After the cooldown has elapsed, another attempt is allowed.
    third = await pipeline.recover_5min_stall(
        status, db_session,
        now=NOW + timedelta(minutes=pipeline.FIVEMIN_RECOVERY_COOLDOWN_MINUTES + 1),
    )
    assert third["attempted"] is True
    assert pipeline.fetcher.fetch_5min_range.await_count == 2


@pytest.mark.asyncio
async def test_incremental_run_triggers_recovery_when_stale(db_session):
    """run_incremental_update wires staleness → recovery."""
    pipeline = IngestionPipeline()
    pipeline.fetcher.fetch_incremental_voo = AsyncMock(
        return_value={"daily": pd.DataFrame(), "5min": pd.DataFrame()}
    )
    pipeline.fetcher.fetch_historical_vix = AsyncMock(return_value=pd.DataFrame())
    pipeline.fetcher.fetch_historical_spx = AsyncMock(return_value=pd.DataFrame())
    pipeline.recover_5min_stall = AsyncMock(
        return_value={"attempted": True, "recovered": True, "bars_fetched": 1, "reason": None}
    )

    # Whether recovery fires depends on live market hours; only assert wiring
    # when the check actually reports stale right now.
    status = await check_5min_staleness(db_session)
    await pipeline.run_incremental_update(db_session)
    if status["stale"]:
        pipeline.recover_5min_stall.assert_awaited_once()
    else:
        pipeline.recover_5min_stall.assert_not_awaited()
