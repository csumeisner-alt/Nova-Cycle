"""Tests for the resampled intraday timeframes on GET /api/voo_candles."""

from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from main import app
from database.db import get_db
from database.models import Base, VooCandle


def _client():
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver/api",
    )


async def _make_db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/tf.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _five_min_candles(start: datetime, n: int, session_type: str = "regular"):
    rows = []
    for i in range(n):
        base = 100.0 + i
        rows.append(
            VooCandle(
                ticker="VOO",
                timestamp=start + timedelta(minutes=5 * i),
                open=base,
                high=base + 2.0,
                low=base - 1.0,
                close=base + 1.0,
                volume=1000 + i,
                timeframe="5min",
                is_extended_hours=session_type != "regular",
                session_type=session_type,
                gap_percent=0.0,
                gap_type=None,
            )
        )
    return rows


@pytest.mark.asyncio
async def test_15min_timeframe_resamples_5min_candles(tmp_path):
    engine, session_factory = await _make_db(tmp_path)
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    async with session_factory() as session:
        session.add_all(_five_min_candles(start, 6))
        await session.commit()

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        async with _client() as client:
            resp = await client.get(
                "/voo_candles", params={"ticker": "VOO", "window": "7d", "timeframe": "15min"}
            )
    finally:
        app.dependency_overrides.pop(get_db, None)
        await engine.dispose()

    assert resp.status_code == 200
    candles = resp.json()
    assert len(candles) == 2  # 6 five-minute bars -> 2 fifteen-minute bars

    first = candles[0]
    # OHLC aggregation: open of first bar, close of last bar in the bucket,
    # max high / min low across the bucket, summed volume.
    assert first["open"] == 100.0
    assert first["close"] == 103.0  # close of the 3rd 5-min bar (102 + 1)
    assert first["high"] == 104.0   # high of the 3rd bar (102 + 2)
    assert first["low"] == 99.0     # low of the 1st bar (100 - 1)
    assert first["volume"] == 1000 + 1001 + 1002
    assert first["timeframe"] == "15min"
    assert first["session_type"] == "regular"


@pytest.mark.asyncio
async def test_1h_timeframe_does_not_blend_sessions(tmp_path):
    engine, session_factory = await _make_db(tmp_path)
    hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    async with session_factory() as session:
        # Same hour bucket, two different sessions: must yield two 1h bars.
        session.add_all(_five_min_candles(hour, 3, session_type="pre_market"))
        session.add_all(
            _five_min_candles(hour + timedelta(minutes=30), 3, session_type="regular")
        )
        await session.commit()

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        async with _client() as client:
            resp = await client.get(
                "/voo_candles", params={"ticker": "VOO", "window": "7d", "timeframe": "1h"}
            )
    finally:
        app.dependency_overrides.pop(get_db, None)
        await engine.dispose()

    assert resp.status_code == 200
    candles = resp.json()
    assert len(candles) == 2
    assert {c["session_type"] for c in candles} == {"pre_market", "regular"}
    pre = next(c for c in candles if c["session_type"] == "pre_market")
    assert pre["is_extended_hours"] is True


@pytest.mark.asyncio
async def test_same_bucket_sessions_keep_chronological_order(tmp_path):
    """Regular + after-hours in the same 1h bucket must stay in trade-time
    order, not lexical session order ('after_hours' < 'regular')."""
    engine, session_factory = await _make_db(tmp_path)
    hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    async with session_factory() as session:
        # Regular session first (:00-:15), then after-hours (:30-:45).
        session.add_all(_five_min_candles(hour, 3, session_type="regular"))
        session.add_all(
            _five_min_candles(hour + timedelta(minutes=30), 3, session_type="after_hours")
        )
        await session.commit()

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        async with _client() as client:
            resp = await client.get(
                "/voo_candles", params={"ticker": "VOO", "window": "7d", "timeframe": "1h"}
            )
    finally:
        app.dependency_overrides.pop(get_db, None)
        await engine.dispose()

    assert resp.status_code == 200
    candles = resp.json()
    assert len(candles) == 2
    assert [c["session_type"] for c in candles] == ["regular", "after_hours"]
    # No internal sort-key leakage into the API payload.
    assert all("_first_ts" not in c for c in candles)


@pytest.mark.asyncio
async def test_unknown_timeframe_rejected(tmp_path):
    engine, session_factory = await _make_db(tmp_path)

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        async with _client() as client:
            resp = await client.get(
                "/voo_candles", params={"ticker": "VOO", "window": "7d", "timeframe": "2min"}
            )
    finally:
        app.dependency_overrides.pop(get_db, None)
        await engine.dispose()

    assert resp.status_code == 400
