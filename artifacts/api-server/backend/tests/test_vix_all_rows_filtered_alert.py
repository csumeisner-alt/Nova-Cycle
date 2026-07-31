"""VIX index volume must not trigger a false degraded-data alert."""

import logging
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, VixCandle
# Note: other test files reload routers.predictions via importlib.reload,
# which would leave `from routers.predictions import X` bindings stale.
# Always access module attributes at runtime instead.
from routers import predictions


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


@pytest.fixture(autouse=True)
def reset_stats():
    """Isolate the module-level counter for each test."""
    predictions._vix_all_filtered_stats.update(count=0, last_at=None, rows_filtered=None)
    yield
    predictions._vix_all_filtered_stats.update(count=0, last_at=None, rows_filtered=None)


def _vix(ts, volume):
    return VixCandle(
        ticker=settings.VIX_TICKER, timestamp=ts, open=1, high=1,
        low=1, close=1, volume=volume, timeframe="daily",
    )


def _recent_days(n):
    now = datetime.utcnow()
    return [now - timedelta(days=i) for i in range(n)]


@pytest.mark.asyncio
async def test_all_zero_volume_rows_are_loaded_without_warning(db_session, caplog):
    for ts in _recent_days(3):
        db_session.add(_vix(ts, volume=0))
    await db_session.flush()

    with caplog.at_level(logging.WARNING, logger="routers.predictions"):
        df = await predictions._load_vix_candles(db_session)

    assert len(df) == 3
    assert not any("vix_prediction_all_rows_filtered" in r.message for r in caplog.records)
    assert predictions._vix_all_filtered_stats["count"] == 0


@pytest.mark.asyncio
async def test_no_warning_when_some_rows_valid(db_session, caplog):
    days = _recent_days(3)
    db_session.add(_vix(days[0], volume=0))
    db_session.add(_vix(days[1], volume=1000))
    await db_session.flush()

    with caplog.at_level(logging.WARNING, logger="routers.predictions"):
        df = await predictions._load_vix_candles(db_session)

    assert len(df) == 2
    assert not any(
        "vix_prediction_all_rows_filtered" in r.message for r in caplog.records
    )
    assert predictions._vix_all_filtered_stats["count"] == 0


@pytest.mark.asyncio
async def test_no_warning_when_table_empty(db_session):
    df = await predictions._load_vix_candles(db_session)
    assert df.empty
    # An empty table is plain missing data (covered by the staleness check),
    # not a zero-volume wipe.
    assert predictions._vix_all_filtered_stats["count"] == 0


@pytest.mark.asyncio
async def test_healthz_surfaces_all_rows_filtered_alert():
    predictions._record_vix_all_rows_filtered(5)

    from httpx import AsyncClient, ASGITransport
    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/healthz")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "degraded"
    assert body["vix_zero_volume_filter"]["count"] == 1
    assert body["vix_zero_volume_filter"]["rows_filtered"] == 5
    assert any(
        a.startswith("vix_prediction_all_rows_filtered") for a in body["alerts"]
    )


@pytest.mark.asyncio
async def test_healthz_clean_when_no_filtering_occurred():
    from httpx import AsyncClient, ASGITransport
    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/healthz")
    assert resp.status_code == 200
    body = resp.json()

    assert body["vix_zero_volume_filter"]["count"] == 0
    assert not any(
        a.startswith("vix_prediction_all_rows_filtered") for a in body["alerts"]
    )
