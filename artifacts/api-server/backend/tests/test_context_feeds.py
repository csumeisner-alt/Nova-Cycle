"""
Tests for the broader-context feed ingestion helpers.

Covers:
  - store_context_candles : duplicate skip, OHLC validation rejection,
                            successful insert
  - check_context_staleness: stale=True when feed lags VOO, stale=False when
                              current, handles missing table rows (no feed data)
  - _ingest_context_tickers : empty ticker is skipped silently, a fetch error
                               for one ticker does not abort the others
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import (
    Base,
    BreadthCandle,
    CreditHyCandle,
    CreditIgCandle,
    RatesCandle,
    VixLongCandle,
    VixShortCandle,
    VooCandle,
)
from ingestion import market_calendar
from ingestion.pipeline import (
    IngestionPipeline,
    check_context_staleness,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _voo(ts):
    return VooCandle(
        ticker=settings.TICKER,
        timestamp=ts,
        open=1,
        high=1,
        low=1,
        close=1,
        volume=1,
        timeframe="daily",
        is_extended_hours=False,
        session_type="regular",
        gap_percent=0.0,
        gap_type="none",
    )


def _recent_trading_days(n):
    """Return the last *n* trading-day datetimes (UTC-naive), oldest first."""
    days, d = [], datetime.utcnow().date()
    while len(days) < n:
        if market_calendar.is_trading_day(d):
            days.append(datetime(d.year, d.month, d.day))
        d -= timedelta(days=1)
    return list(reversed(days))


def _make_candle_df(rows):
    """Build a minimal OHLCV DataFrame suitable for store_context_candles.

    *rows* is a list of (datetime, open, high, low, close, volume) tuples.
    """
    index = [pd.Timestamp(ts) for ts, *_ in rows]
    data = {
        "open":   [o for _, o, *_ in rows],
        "high":   [h for _, _, h, *_ in rows],
        "low":    [l for _, _, _, l, *_ in rows],
        "close":  [c for _, _, _, _, c, *_ in rows],
        "volume": [v for *_, v in rows],
    }
    return pd.DataFrame(data, index=index)


# ─────────────────────────────────────────────────────────────────────────────
# store_context_candles
# ─────────────────────────────────────────────────────────────────────────────


class TestStoreContextCandles:
    """Unit-test the generic store_context_candles helper."""

    @pytest.mark.asyncio
    async def test_successful_insert(self, db_session):
        """Fresh rows are inserted and counted correctly."""
        pipeline = IngestionPipeline()
        ts = datetime(2024, 3, 1)
        df = _make_candle_df([(ts, 20.0, 22.0, 19.5, 21.0, 0.0)])

        await pipeline.store_context_candles(
            df,
            db_session,
            model=VixShortCandle,
            ticker=settings.VIX_SHORT_TICKER,
            label="VIX9D",
        )

        from sqlalchemy import select, func
        result = await db_session.execute(
            select(func.count(VixShortCandle.id)).where(
                VixShortCandle.ticker == settings.VIX_SHORT_TICKER
            )
        )
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_duplicate_skip(self, db_session):
        """A candle with the same timestamp is skipped on the second call."""
        pipeline = IngestionPipeline()
        ts = datetime(2024, 3, 1)
        df = _make_candle_df([(ts, 20.0, 22.0, 19.5, 21.0, 0.0)])

        # Insert once
        await pipeline.store_context_candles(
            df,
            db_session,
            model=VixShortCandle,
            ticker=settings.VIX_SHORT_TICKER,
            label="VIX9D",
        )

        # Insert again with the same timestamp
        await pipeline.store_context_candles(
            df,
            db_session,
            model=VixShortCandle,
            ticker=settings.VIX_SHORT_TICKER,
            label="VIX9D",
        )

        from sqlalchemy import select, func
        result = await db_session.execute(
            select(func.count(VixShortCandle.id)).where(
                VixShortCandle.ticker == settings.VIX_SHORT_TICKER
            )
        )
        # Exactly one row despite two store calls
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_ohlc_validation_rejection(self, db_session):
        """A candle whose high < open is rejected and not stored."""
        pipeline = IngestionPipeline()
        ts = datetime(2024, 3, 1)
        # high (10) < open (20) → invalid OHLC
        df = _make_candle_df([(ts, 20.0, 10.0, 9.0, 15.0, 0.0)])

        await pipeline.store_context_candles(
            df,
            db_session,
            model=VixShortCandle,
            ticker=settings.VIX_SHORT_TICKER,
            label="VIX9D",
        )

        from sqlalchemy import select, func
        result = await db_session.execute(
            select(func.count(VixShortCandle.id)).where(
                VixShortCandle.ticker == settings.VIX_SHORT_TICKER
            )
        )
        assert result.scalar() == 0

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid_rows(self, db_session):
        """Only the rows that pass OHLC validation are stored."""
        pipeline = IngestionPipeline()
        ts_valid = datetime(2024, 3, 1)
        ts_bad = datetime(2024, 3, 4)
        df = _make_candle_df([
            (ts_valid, 20.0, 22.0, 19.0, 21.0, 100.0),   # valid
            (ts_bad,   20.0,  5.0,  4.0, 15.0,   0.0),   # high < open → invalid
        ])

        await pipeline.store_context_candles(
            df,
            db_session,
            model=VixShortCandle,
            ticker=settings.VIX_SHORT_TICKER,
            label="VIX9D",
        )

        from sqlalchemy import select, func
        result = await db_session.execute(
            select(func.count(VixShortCandle.id)).where(
                VixShortCandle.ticker == settings.VIX_SHORT_TICKER
            )
        )
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_empty_dataframe_is_noop(self, db_session):
        """Passing an empty DataFrame leaves the table unchanged."""
        pipeline = IngestionPipeline()
        df = pd.DataFrame()

        # Must not raise
        await pipeline.store_context_candles(
            df,
            db_session,
            model=VixShortCandle,
            ticker=settings.VIX_SHORT_TICKER,
            label="VIX9D",
        )

        from sqlalchemy import select, func
        result = await db_session.execute(
            select(func.count(VixShortCandle.id))
        )
        assert result.scalar() == 0

    @pytest.mark.asyncio
    async def test_negative_volume_is_rejected(self, db_session):
        """A row with negative volume is skipped."""
        pipeline = IngestionPipeline()
        ts = datetime(2024, 3, 1)
        df = _make_candle_df([(ts, 20.0, 22.0, 19.0, 21.0, -5.0)])

        await pipeline.store_context_candles(
            df,
            db_session,
            model=VixShortCandle,
            ticker=settings.VIX_SHORT_TICKER,
            label="VIX9D",
        )

        from sqlalchemy import select, func
        result = await db_session.execute(
            select(func.count(VixShortCandle.id))
        )
        assert result.scalar() == 0


# ─────────────────────────────────────────────────────────────────────────────
# check_context_staleness
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckContextStaleness:
    """Unit-test check_context_staleness for all six context feeds."""

    @pytest.mark.asyncio
    async def test_no_voo_data_never_stale(self, db_session):
        """Without any VOO reference data all feeds report stale=False."""
        results = await check_context_staleness(db_session)
        assert isinstance(results, list)
        assert len(results) == 6
        for status in results:
            assert status["stale"] is False
            assert status["latest_voo"] is None

    @pytest.mark.asyncio
    async def test_missing_feed_with_voo_is_stale(self, db_session):
        """Feed with no rows at all while VOO data exists → stale=True."""
        days = _recent_trading_days(3)
        for ts in days:
            db_session.add(_voo(ts))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        # Every feed should be stale (no context candles ingested)
        for status in results:
            assert status["stale"] is True, (
                f"Expected stale for feed with key {status}"
            )

    @pytest.mark.asyncio
    async def test_fresh_feed_not_stale(self, db_session):
        """All six feeds current with VOO → stale=False, lag==0."""
        days = _recent_trading_days(5)
        for ts in days:
            db_session.add(_voo(ts))
            db_session.add(VixShortCandle(
                ticker=settings.VIX_SHORT_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
            ))
            db_session.add(VixLongCandle(
                ticker=settings.VIX_LONG_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
            ))
            db_session.add(RatesCandle(
                ticker=settings.RATES_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
            ))
            db_session.add(CreditHyCandle(
                ticker=settings.CREDIT_HY_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=100.0, timeframe="daily",
            ))
            db_session.add(CreditIgCandle(
                ticker=settings.CREDIT_IG_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=100.0, timeframe="daily",
            ))
            db_session.add(BreadthCandle(
                ticker=settings.BREADTH_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
            ))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        for status in results:
            assert status["stale"] is False
            assert status["lag_trading_days"] == 0

    @pytest.mark.asyncio
    async def test_lagging_feed_is_stale(self, db_session):
        """A feed that stopped updating exceeds the lag threshold → stale."""
        n = settings.LONG_CONTEXT_STALENESS_MAX_DAYS + 5
        days = _recent_trading_days(n)
        for ts in days:
            db_session.add(_voo(ts))
        # Only the oldest day has a VIX9D candle; all newer ones are missing
        db_session.add(VixShortCandle(
            ticker=settings.VIX_SHORT_TICKER, timestamp=days[0],
            open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
        ))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        # Find the VIX9D entry ("latest_vix_short" is the dict key)
        vix_short_matches = [s for s in results if "latest_vix_short" in s]
        assert vix_short_matches, "Expected a vix_short status dict in results"
        vix_short = vix_short_matches[0]
        assert vix_short["stale"] is True
        assert vix_short["lag_trading_days"] > settings.LONG_CONTEXT_STALENESS_MAX_DAYS
        assert "lags" in vix_short["detail"].lower() or "lag" in vix_short["detail"].lower()

    @pytest.mark.asyncio
    async def test_lag_within_threshold_not_stale(self, db_session):
        """A feed lagging by exactly the allowed days is not stale."""
        n = settings.LONG_CONTEXT_STALENESS_MAX_DAYS + 3
        days = _recent_trading_days(n)
        for ts in days:
            db_session.add(_voo(ts))
        # Feed present at index -(max_days+1), so lag == max_days exactly
        cutoff_idx = -(settings.LONG_CONTEXT_STALENESS_MAX_DAYS + 1)
        db_session.add(VixShortCandle(
            ticker=settings.VIX_SHORT_TICKER, timestamp=days[cutoff_idx],
            open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
        ))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        vix_short_matches = [s for s in results if "latest_vix_short" in s]
        assert vix_short_matches, "Expected a vix_short status dict in results"
        vix_short = vix_short_matches[0]
        assert vix_short["stale"] is False
        assert vix_short["lag_trading_days"] == settings.LONG_CONTEXT_STALENESS_MAX_DAYS


# ─────────────────────────────────────────────────────────────────────────────
# _ingest_context_tickers
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestContextTickers:
    """Unit-test IngestionPipeline._ingest_context_tickers behaviour."""

    @pytest.mark.asyncio
    async def test_empty_ticker_is_silently_skipped(self, db_session):
        """When a ticker config value is empty, that feed is skipped without error."""
        pipeline = IngestionPipeline()

        fetch_mock = AsyncMock(return_value=pd.DataFrame())

        with (
            patch.object(settings, "VIX_SHORT_TICKER", ""),
            patch.object(pipeline.fetcher, "fetch_historical_context_ticker", fetch_mock),
        ):
            # Must not raise; empty ticker must not call the fetcher
            await pipeline._ingest_context_tickers(db_session, years=1)

        # The empty ticker should never have been passed to the fetcher
        for call_args in fetch_mock.call_args_list:
            assert call_args.args[0] != "", "Empty ticker should not be fetched"

    @pytest.mark.asyncio
    async def test_fetch_error_for_one_ticker_does_not_abort_others(self, db_session):
        """A fetch exception for ticker N must not prevent ticker N+1 from being ingested."""
        pipeline = IngestionPipeline()

        ts = datetime(2024, 3, 1)
        valid_df = _make_candle_df([(ts, 20.0, 22.0, 19.0, 21.0, 0.0)])

        call_count = 0

        async def _fetch_side_effect(ticker, *, years, is_index):
            nonlocal call_count
            call_count += 1
            # Fail on the first ticker (VIX9D), succeed for all others
            if ticker == settings.VIX_SHORT_TICKER:
                raise RuntimeError("simulated vendor failure")
            return valid_df

        with patch.object(
            pipeline.fetcher, "fetch_historical_context_ticker", _fetch_side_effect
        ):
            # Must not raise even though the first ticker throws
            await pipeline._ingest_context_tickers(db_session, years=1)

        # All 6 tickers should have been attempted
        assert call_count == 6

        # The remaining 5 valid tickers should have inserted rows
        from sqlalchemy import select, func

        for model, attr in [
            (VixLongCandle, "VIX_LONG_TICKER"),
            (RatesCandle, "RATES_TICKER"),
            (CreditHyCandle, "CREDIT_HY_TICKER"),
            (CreditIgCandle, "CREDIT_IG_TICKER"),
            (BreadthCandle, "BREADTH_TICKER"),
        ]:
            ticker_val = getattr(settings, attr)
            result = await db_session.execute(
                select(func.count(model.id)).where(model.ticker == ticker_val)
            )
            assert result.scalar() == 1, (
                f"Expected 1 row for {attr}={ticker_val} after other tickers failed"
            )

        # The failing ticker (VIX9D) should have no rows
        result = await db_session.execute(
            select(func.count(VixShortCandle.id)).where(
                VixShortCandle.ticker == settings.VIX_SHORT_TICKER
            )
        )
        assert result.scalar() == 0

    @pytest.mark.asyncio
    async def test_all_tickers_fetched_successfully(self, db_session):
        """When every ticker fetches cleanly all six feeds are stored."""
        pipeline = IngestionPipeline()

        ts = datetime(2024, 3, 1)
        valid_df = _make_candle_df([(ts, 20.0, 22.0, 19.0, 21.0, 0.0)])

        with patch.object(
            pipeline.fetcher,
            "fetch_historical_context_ticker",
            AsyncMock(return_value=valid_df),
        ):
            await pipeline._ingest_context_tickers(db_session, years=1)

        from sqlalchemy import select, func

        for model, attr in [
            (VixShortCandle, "VIX_SHORT_TICKER"),
            (VixLongCandle, "VIX_LONG_TICKER"),
            (RatesCandle, "RATES_TICKER"),
            (CreditHyCandle, "CREDIT_HY_TICKER"),
            (CreditIgCandle, "CREDIT_IG_TICKER"),
            (BreadthCandle, "BREADTH_TICKER"),
        ]:
            ticker_val = getattr(settings, attr)
            result = await db_session.execute(
                select(func.count(model.id)).where(model.ticker == ticker_val)
            )
            assert result.scalar() == 1, f"Expected 1 row for {attr}"

    @pytest.mark.asyncio
    async def test_result_includes_ticker_and_feed_key(self, db_session):
        """Each entry returned by check_context_staleness has ticker and feed_key."""
        results = await check_context_staleness(db_session)
        assert len(results) == 6
        for status in results:
            assert "ticker" in status, f"Missing 'ticker' in {status}"
            assert "feed_key" in status, f"Missing 'feed_key' in {status}"
            assert status["ticker"], "ticker must be a non-empty string"
            assert status["feed_key"], "feed_key must be a non-empty string"

    @pytest.mark.asyncio
    async def test_empty_fetch_result_warns_and_continues(self, db_session):
        """When the fetcher returns an empty DataFrame no rows are inserted."""
        pipeline = IngestionPipeline()

        with patch.object(
            pipeline.fetcher,
            "fetch_historical_context_ticker",
            AsyncMock(return_value=pd.DataFrame()),
        ):
            # Must not raise
            await pipeline._ingest_context_tickers(db_session, years=1)

        from sqlalchemy import select, func

        for model in (
            VixShortCandle, VixLongCandle, RatesCandle,
            CreditHyCandle, CreditIgCandle, BreadthCandle,
        ):
            result = await db_session.execute(select(func.count(model.id)))
            assert result.scalar() == 0


# ─────────────────────────────────────────────────────────────────────────────
# /api/healthz context_feeds integration
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthzContextFeeds:
    """Verify that /api/healthz surfaces context-feed staleness correctly."""

    @pytest.mark.asyncio
    async def test_healthz_includes_context_feeds_block(self, db_session):
        """/healthz response always contains a context_feeds list."""
        from routers.predictions import healthz

        body = await healthz(session=db_session)
        assert "context_feeds" in body
        assert isinstance(body["context_feeds"], list)

    @pytest.mark.asyncio
    async def test_healthz_context_feeds_stale_sets_degraded_and_alert(self, db_session):
        """When a feed is stale, healthz status is degraded and alerts contains the feed."""
        from routers.predictions import healthz

        # Insert VOO data so staleness checks have a reference date.
        days = _recent_trading_days(3)
        for ts in days:
            db_session.add(_voo(ts))
        await db_session.flush()
        # No context candles → every feed should be stale.

        body = await healthz(session=db_session)

        assert body["status"] == "degraded", (
            "Expected status=degraded when context feeds are stale"
        )
        # At least one context_feed alert must appear.
        context_alerts = [a for a in body["alerts"] if a.startswith("context_feed ")]
        assert context_alerts, (
            f"Expected at least one 'context_feed ...' alert; got alerts={body['alerts']}"
        )

    @pytest.mark.asyncio
    async def test_healthz_fresh_feeds_no_context_alert(self, db_session):
        """When all six feeds are current there are no context_feed alerts."""
        from routers.predictions import healthz

        days = _recent_trading_days(5)
        for ts in days:
            db_session.add(_voo(ts))
            db_session.add(VixShortCandle(
                ticker=settings.VIX_SHORT_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
            ))
            db_session.add(VixLongCandle(
                ticker=settings.VIX_LONG_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
            ))
            db_session.add(RatesCandle(
                ticker=settings.RATES_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
            ))
            db_session.add(CreditHyCandle(
                ticker=settings.CREDIT_HY_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=100.0, timeframe="daily",
            ))
            db_session.add(CreditIgCandle(
                ticker=settings.CREDIT_IG_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=100.0, timeframe="daily",
            ))
            db_session.add(BreadthCandle(
                ticker=settings.BREADTH_TICKER, timestamp=ts,
                open=1, high=1, low=1, close=1, volume=0.0, timeframe="daily",
            ))
        await db_session.flush()

        body = await healthz(session=db_session)

        context_alerts = [a for a in body["alerts"] if a.startswith("context_feed ")]
        assert not context_alerts, (
            f"Expected no context_feed alerts when all feeds are fresh; got {context_alerts}"
        )
        # All six feeds should report stale=False.
        for feed in body["context_feeds"]:
            assert feed["stale"] is False, f"Expected stale=False for {feed.get('ticker')}"

    @pytest.mark.asyncio
    async def test_healthz_context_feeds_carry_ticker_and_feed_key(self, db_session):
        """Each entry in context_feeds has non-empty ticker and feed_key fields."""
        from routers.predictions import healthz

        body = await healthz(session=db_session)
        feeds = body["context_feeds"]
        assert len(feeds) == 6, f"Expected 6 context feeds, got {len(feeds)}"
        for feed in feeds:
            assert feed.get("ticker"), f"Missing ticker in {feed}"
            assert feed.get("feed_key"), f"Missing feed_key in {feed}"
