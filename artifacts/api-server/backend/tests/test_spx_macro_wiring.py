"""Tests for SPX futures wiring into the macro sensitivity signal.

Covers:
  - compute_macro_sensitivity uses the real SPX series when provided
    (no fallback log) and falls back to the VOO proxy when it is absent
  - ModelTrainer._load_spx_close returns the stored series / empty on no data
  - the prediction router's _load_spx_close_series helper loads the series
"""

import logging

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from database.models import Base, SpxCandle
from config import settings
from ml import features as ml_features
from ml.trainer import ModelTrainer


# ── Feature-level: real series vs fallback ───────────────────────────────────

def _series(n=40, start=100.0, seed=1):
    idx = pd.bdate_range("2026-05-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(start + np.cumsum(rng.normal(0, 1.0, n)), index=idx)


def test_macro_sensitivity_uses_spx_when_available(caplog):
    close = _series()
    spx = _series(start=5000.0, seed=2)
    with caplog.at_level(logging.INFO, logger="ml.features"):
        score = ml_features.compute_macro_sensitivity(
            close, open_=close * 1.001, spx_futures_close=spx
        )
    assert score.between(0, 1).all()
    assert "spx_futures_unavailable" not in caplog.text


def test_macro_sensitivity_falls_back_without_spx(caplog):
    close = _series()
    with caplog.at_level(logging.INFO, logger="ml.features"):
        score = ml_features.compute_macro_sensitivity(close, open_=close * 1.001)
    assert score.between(0, 1).all()
    assert "spx_futures_unavailable" in caplog.text


def test_spx_changes_the_score():
    close = _series()
    # Highly volatile SPX series should raise the score vs a flat one
    idx = close.index
    flat_spx = pd.Series(5000.0, index=idx)
    wild_spx = pd.Series(5000 * (1 + 0.03 * np.sin(np.arange(len(idx)))), index=idx)
    s_flat = ml_features.compute_macro_sensitivity(close, spx_futures_close=flat_spx)
    s_wild = ml_features.compute_macro_sensitivity(close, spx_futures_close=wild_spx)
    assert s_wild.mean() > s_flat.mean()


# ── DB-level: trainer + router loaders ───────────────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


async def _seed_spx(session, n=10):
    for i, ts in enumerate(pd.bdate_range("2026-07-01", periods=n)):
        session.add(SpxCandle(
            ticker=settings.SPX_FUTURES_TICKER,
            timestamp=ts.to_pydatetime(),
            open=5000.0 + i, high=5010.0 + i, low=4990.0 + i,
            close=5005.0 + i, volume=1000.0, timeframe="daily",
        ))
    await session.flush()


@pytest.mark.asyncio
async def test_trainer_loads_spx_close(db_session):
    await _seed_spx(db_session, n=10)
    series = await ModelTrainer._load_spx_close(db_session)
    assert len(series) == 10
    assert float(series.iloc[-1]) == pytest.approx(5014.0)
    assert series.index.is_monotonic_increasing


@pytest.mark.asyncio
async def test_trainer_returns_empty_series_without_spx(db_session):
    series = await ModelTrainer._load_spx_close(db_session)
    assert series.empty


@pytest.mark.asyncio
async def test_prediction_router_loads_spx_close(db_session):
    from routers.predictions import _load_spx_close_series
    await _seed_spx(db_session, n=5)
    series = await _load_spx_close_series(db_session)
    assert len(series) == 5
    assert float(series.iloc[-1]) == pytest.approx(5009.0)


def test_align_spx_to_integer_indexed_df():
    from routers.predictions import _align_spx_to_df
    spx = pd.Series(
        [5000.0, 5010.0, 5020.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
    )
    # Router-style frame: integer index + timestamp column (intraday rows)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-07-02 13:30", "2026-07-02 13:35", "2026-07-03 13:30",
        ]),
        "close": [100.0, 100.1, 100.2],
    })
    aligned = _align_spx_to_df(spx, df)
    assert list(aligned.index) == [0, 1, 2]
    assert list(aligned.values) == [5010.0, 5010.0, 5020.0]
    # And the feature layer accepts the aligned series without erroring
    score = ml_features.compute_macro_sensitivity(
        df["close"], spx_futures_close=aligned
    )
    assert score.between(0, 1).all()


@pytest.mark.asyncio
async def test_prediction_router_empty_without_spx(db_session):
    from routers.predictions import _load_spx_close_series
    series = await _load_spx_close_series(db_session)
    assert series.empty
