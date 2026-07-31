"""
Tests: zero-volume SPX bars must not silently corrupt the macro signal.

remove_invalid_spx_candles() removes bad SPX rows at startup, but a
zero-volume row may survive from an older backup restore or a race before
cleanup completes.  The SPX read path in the prediction router must filter
such rows itself rather than letting a bad close value reach the
overnight-return / macro-sensitivity signal.
"""

from datetime import datetime

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, SpxCandle


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _spx_candle(ts: datetime, *, close: float = 5000.0, volume: float | None = 1.0) -> SpxCandle:
    """Build a minimal SpxCandle row for testing."""
    return SpxCandle(
        ticker=settings.SPX_FUTURES_TICKER,
        timestamp=ts,
        open=close,
        high=close + 10,
        low=close - 10,
        close=close,
        volume=volume,
        timeframe="daily",
    )


# ---------------------------------------------------------------------------
# Unit tests for _load_spx_close_series zero-volume guard
# ---------------------------------------------------------------------------

class TestLoadSpxCloseSeriesZeroVolumeGuard:
    """_load_spx_close_series must skip zero-volume rows silently."""

    @pytest.mark.asyncio
    async def test_normal_spx_row_is_returned(self, db_session):
        """A valid SPX row (volume > 0) is loaded and returned."""
        from routers.predictions import _load_spx_close_series

        db_session.add(_spx_candle(datetime(2026, 7, 28), close=5100.0, volume=1.0))
        await db_session.flush()

        series = await _load_spx_close_series(db_session, limit=10)
        assert not series.empty
        assert len(series) == 1
        assert float(series.iloc[0]) == pytest.approx(5100.0)

    @pytest.mark.asyncio
    async def test_zero_volume_row_is_skipped(self, db_session):
        """An SPX row with volume=0 is excluded from the returned series."""
        from routers.predictions import _load_spx_close_series

        db_session.add(_spx_candle(datetime(2026, 7, 25), close=1.0, volume=0.0))
        await db_session.flush()

        series = await _load_spx_close_series(db_session, limit=10)
        assert series.empty, "zero-volume SPX row should have been skipped"

    @pytest.mark.asyncio
    async def test_null_volume_row_is_skipped(self, db_session):
        """An SPX row with volume=None (NULL in DB) is excluded."""
        from routers.predictions import _load_spx_close_series

        db_session.add(_spx_candle(datetime(2026, 7, 25), close=1.0, volume=None))
        await db_session.flush()

        series = await _load_spx_close_series(db_session, limit=10)
        assert series.empty, "null-volume SPX row should have been skipped"

    @pytest.mark.asyncio
    async def test_zero_volume_row_does_not_pollute_valid_rows(self, db_session):
        """A mix of valid and zero-volume rows: only valid closes are returned."""
        from routers.predictions import _load_spx_close_series

        ts_good_1 = datetime(2026, 7, 24)
        ts_bad    = datetime(2026, 7, 25)   # zero-volume glitch
        ts_good_2 = datetime(2026, 7, 28)

        db_session.add(_spx_candle(ts_good_1, close=5000.0, volume=1.0))
        db_session.add(_spx_candle(ts_bad,    close=1.0,    volume=0.0))
        db_session.add(_spx_candle(ts_good_2, close=5050.0, volume=1.0))
        await db_session.flush()

        series = await _load_spx_close_series(db_session, limit=10)

        assert len(series) == 2, f"expected 2 valid rows, got {len(series)}"
        closes = list(series.astype(float))
        assert 1.0 not in closes, "zero-volume close must be excluded from result"
        assert closes == [5000.0, 5050.0]
        # Index must stay chronological with the bad timestamp removed
        assert list(series.index) == [pd.Timestamp(ts_good_1), pd.Timestamp(ts_good_2)]

    @pytest.mark.asyncio
    async def test_wrong_ticker_rows_are_ignored(self, db_session):
        """Rows for a different ticker never enter the series."""
        from routers.predictions import _load_spx_close_series

        other = _spx_candle(datetime(2026, 7, 28), close=400.0, volume=1.0)
        other.ticker = "VOO"
        db_session.add(other)
        await db_session.flush()

        series = await _load_spx_close_series(db_session, limit=10)
        assert series.empty
