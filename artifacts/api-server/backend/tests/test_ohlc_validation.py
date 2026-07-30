"""Regression coverage for malformed vendor OHLC rows and repairs."""

from datetime import datetime

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.models import Base, VooCandle
from ingestion.fetcher import DataFetcher, ohlc_validation_issue
from ingestion.pipeline import IngestionPipeline


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def test_impossible_high_is_rejected():
    assert ohlc_validation_issue(680.12, 676.71, 675.58, 676.01) == (
        "high_below_open_or_close"
    )


def test_valid_ohlc_is_accepted():
    assert ohlc_validation_issue(676.54, 682.44, 675.22, 681.79) is None


def test_normalise_columns_drops_bad_row():
    frame = pd.DataFrame(
        {
            "Open": [680.12, 676.54],
            "High": [676.71, 682.44],
            "Low": [675.58, 675.22],
            "Close": [676.01, 681.79],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2026-07-30", "2026-07-31"]),
    )

    result = DataFetcher._normalise_columns(frame)

    assert list(result.index) == [pd.Timestamp("2026-07-31")]


@pytest.mark.asyncio
async def test_existing_bad_row_is_repaired(db_session):
    bad = VooCandle(
        ticker="VOO",
        timestamp=datetime(2026, 7, 30),
        open=680.12,
        high=676.71,
        low=675.58,
        close=676.01,
        volume=100,
        timeframe="daily",
        session_type="regular",
    )
    db_session.add(bad)
    await db_session.flush()

    frame = pd.DataFrame(
        {
            "open": [676.54],
            "high": [682.44],
            "low": [675.22],
            "close": [681.79],
            "volume": [5653980],
            "is_extended_hours": [False],
            "session_type": ["regular"],
        },
        index=pd.to_datetime(["2026-07-30"]),
    )

    await IngestionPipeline().store_voo_candles(frame, db_session, "daily")

    await db_session.refresh(bad)
    assert (bad.open, bad.high, bad.low, bad.close) == (
        676.54,
        682.44,
        675.22,
        681.79,
    )


@pytest.mark.asyncio
async def test_startup_cleanup_removes_existing_bad_row(db_session):
    bad = VooCandle(
        ticker="VOO",
        timestamp=datetime(2026, 7, 30),
        open=680.12,
        high=676.71,
        low=675.58,
        close=676.01,
        volume=100,
        timeframe="daily",
        session_type="regular",
    )
    db_session.add(bad)
    await db_session.flush()

    removed = await IngestionPipeline().remove_invalid_voo_candles(db_session)

    assert removed == 1
    assert await db_session.get(VooCandle, bad.id) is None