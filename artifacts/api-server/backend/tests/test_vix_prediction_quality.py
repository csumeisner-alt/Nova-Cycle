"""Regression tests for VIX index candles with non-trading volume."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VixCandle


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


def _vix_candle(ts: datetime, *, close: float = 20.0, volume: float | None = 1.0) -> VixCandle:
    """Build a minimal VixCandle row for testing."""
    return VixCandle(
        ticker=settings.VIX_TICKER,
        timestamp=ts,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=volume,
        timeframe="daily",
    )


# ---------------------------------------------------------------------------
# Unit tests for _load_vix_candles
# ---------------------------------------------------------------------------

class TestLoadVixCandles:
    """_load_vix_candles must not treat VIX index volume as trade volume."""

    @pytest.mark.asyncio
    async def test_normal_vix_row_is_returned(self, db_session):
        """A valid VIX row (volume > 0) is loaded and returned."""
        from routers.predictions import _load_vix_candles

        db_session.add(_vix_candle(datetime(2026, 7, 28), close=18.5, volume=1.0))
        await db_session.flush()

        df = await _load_vix_candles(db_session, limit=10)
        assert not df.empty
        assert len(df) == 1
        assert float(df.iloc[0]["close"]) == pytest.approx(18.5)

    @pytest.mark.asyncio
    async def test_zero_volume_row_is_returned(self, db_session):
        """A valid VIX row with volume=0 is included in the returned frame."""
        from routers.predictions import _load_vix_candles

        db_session.add(_vix_candle(datetime(2026, 7, 25), close=20.0, volume=0.0))
        await db_session.flush()

        df = await _load_vix_candles(db_session, limit=10)
        assert len(df) == 1

    @pytest.mark.asyncio
    async def test_null_volume_row_is_returned(self, db_session):
        """A valid VIX row with volume=NULL is still usable."""
        from routers.predictions import _load_vix_candles

        db_session.add(_vix_candle(datetime(2026, 7, 25), close=20.0, volume=None))
        await db_session.flush()

        df = await _load_vix_candles(db_session, limit=10)
        assert len(df) == 1

    @pytest.mark.asyncio
    async def test_zero_volume_row_is_returned_with_valid_rows(self, db_session):
        """A mix of valid and zero-volume rows is returned in timestamp order."""
        from routers.predictions import _load_vix_candles

        ts_good_1 = datetime(2026, 7, 24)
        ts_zero   = datetime(2026, 7, 25)   # normal index volume
        ts_good_2 = datetime(2026, 7, 28)

        db_session.add(_vix_candle(ts_good_1, close=17.0, volume=1.0))
        db_session.add(_vix_candle(ts_zero,   close=18.0, volume=0.0))
        db_session.add(_vix_candle(ts_good_2, close=19.0, volume=1.0))
        await db_session.flush()

        df = await _load_vix_candles(db_session, limit=10)

        assert len(df) == 3, f"expected 3 VIX rows, got {len(df)}"
        closes = list(df["close"].astype(float))
        assert pytest.approx(17.0) in closes
        assert pytest.approx(18.0) in closes
        assert pytest.approx(19.0) in closes

    @pytest.mark.asyncio
    async def test_all_zero_volume_returns_all_valid_rows(self, db_session):
        """When every VIX row has zero volume, all valid rows are returned."""
        from routers.predictions import _load_vix_candles

        for i in range(3):
            db_session.add(
                _vix_candle(datetime(2026, 7, 21) + timedelta(days=i), close=15.0, volume=0.0)
            )
        await db_session.flush()

        df = await _load_vix_candles(db_session, limit=10)
        assert len(df) == 3

    @pytest.mark.asyncio
    async def test_malformed_zero_volume_row_never_reaches_macro_signal(self, db_session):
        """
        A malformed OHLC row must be rejected even when its volume is zero.

        This covers the race/restore scenario described in the task: the row
        exists in the DB but remove_invalid_vix_candles() has not run yet.
        """
        from routers.predictions import _load_vix_candles

        # Valid row: close=20 (normal VIX level)
        db_session.add(_vix_candle(datetime(2026, 7, 28), close=20.0, volume=1.0))
        # Malformed row: high below close, with normal index volume=0.
        bad = _vix_candle(datetime(2026, 7, 29), close=20.0, volume=0.0)
        bad.high = 19.0
        db_session.add(bad)
        await db_session.flush()

        df = await _load_vix_candles(db_session, limit=10)

        # Only the valid row should be present.
        assert len(df) == 1
        assert float(df.iloc[0]["close"]) == pytest.approx(20.0), (
            "malformed zero-volume VIX candle must not reach the macro signal"
        )

    @pytest.mark.asyncio
    async def test_invalid_ohlc_is_logged(self, db_session, caplog):
        """Skipping malformed VIX OHLC emits a WARNING."""
        import logging
        from routers.predictions import _load_vix_candles

        bad = _vix_candle(datetime(2026, 7, 25), close=30.0, volume=0.0)
        bad.high = 29.0
        db_session.add(bad)
        await db_session.flush()

        with caplog.at_level(logging.WARNING):
            await _load_vix_candles(db_session, limit=10)

        assert any(
            "vix_prediction_invalid_ohlc_skipped" in record.message
            for record in caplog.records
        ), "Expected a WARNING log entry for malformed VIX OHLC"
