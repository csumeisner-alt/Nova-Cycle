"""
Tests for the startup session-label repair pass
(`database.maintenance.reclassify_session_labels`).

Uses an in-memory SQLite database. Verifies that:
  - rows with wrong session_type / is_extended_hours are corrected
  - correct rows are untouched
  - daily candles are ignored
  - the pass is idempotent (second run corrects 0 rows)
"""

import logging
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.maintenance import reclassify_session_labels
from database.models import Base, VooCandle


def _candle(ts, session_type, is_ext, timeframe="5min"):
    return VooCandle(
        ticker="VOO", timestamp=ts,
        open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0,
        timeframe=timeframe, session_type=session_type,
        is_extended_hours=is_ext, gap_type="none",
    )


@pytest_asyncio.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.mark.asyncio
async def test_repairs_wrong_labels_and_is_idempotent(session_maker, caplog):
    async with session_maker() as session:
        session.add_all([
            # 2026-01-15 14:30 UTC == 09:30 EST → regular; stored wrong
            # (a fixed -4 offset would have called it pre_market)
            _candle(datetime(2026, 1, 15, 14, 30), "pre_market", True),
            # Good Friday 2026-04-03 15:00 UTC → holiday → after_hours/extended
            _candle(datetime(2026, 4, 3, 15, 0), "regular", False),
            # Half-day: 2026-11-27 18:00 UTC == 13:00 EST → after_hours
            _candle(datetime(2026, 11, 27, 18, 0), "regular", False),
            # Already correct: 2026-06-15 14:00 UTC == 10:00 EDT → regular
            _candle(datetime(2026, 6, 15, 14, 0), "regular", False),
            # Daily candle with "wrong" label must be ignored
            _candle(datetime(2026, 1, 15, 14, 30), "pre_market", True,
                    timeframe="daily"),
        ])
        await session.commit()

    async with session_maker() as session:
        with caplog.at_level(logging.INFO, logger="database.maintenance"):
            corrected = await reclassify_session_labels(session)
    assert corrected == 3
    assert any("corrected=3" in r.message for r in caplog.records)

    async with session_maker() as session:
        rows = (await session.execute(
            select(VooCandle).order_by(VooCandle.id)
        )).scalars().all()

    assert (rows[0].session_type, rows[0].is_extended_hours) == ("regular", False)
    assert (rows[1].session_type, rows[1].is_extended_hours) == ("after_hours", True)
    assert (rows[2].session_type, rows[2].is_extended_hours) == ("after_hours", True)
    assert (rows[3].session_type, rows[3].is_extended_hours) == ("regular", False)
    # daily row untouched
    assert (rows[4].session_type, rows[4].is_extended_hours) == ("pre_market", True)

    # Idempotent: second pass corrects nothing
    async with session_maker() as session:
        assert await reclassify_session_labels(session) == 0


@pytest.mark.asyncio
async def test_empty_db_corrects_zero(session_maker):
    async with session_maker() as session:
        assert await reclassify_session_labels(session) == 0
