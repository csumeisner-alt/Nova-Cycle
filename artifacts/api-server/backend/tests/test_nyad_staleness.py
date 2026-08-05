"""
Tests confirming the NYAD (NYSE A/D) breadth feed is flagged as stale in the
dashboard when Yahoo Finance stops returning data.

Covers two failure modes:
  1. Missing feed – BreadthCandle table is empty while VOO data exists
     (matches the live YFPricesMissingError / 404 scenario).
  2. Lagging feed – BreadthCandle rows exist but are older than
     LONG_CONTEXT_STALENESS_MAX_DAYS trading days behind VOO.

Both scenarios are exercised at two levels:
  • check_context_staleness() – the pipeline helper that drives the detection
  • /api/healthz – the endpoint that surfaces context_feeds to operators
"""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings
from database.models import Base, BreadthCandle, VooCandle
from ingestion import market_calendar
from ingestion.pipeline import check_context_staleness


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _voo(ts: datetime) -> VooCandle:
    return VooCandle(
        ticker=settings.TICKER,
        timestamp=ts,
        open=1, high=1, low=1, close=1,
        volume=1_000_000.0,
        timeframe="daily",
        is_extended_hours=False,
        session_type="regular",
        gap_percent=0.0,
        gap_type="none",
    )


def _breadth(ts: datetime) -> BreadthCandle:
    return BreadthCandle(
        ticker=settings.BREADTH_TICKER,
        timestamp=ts,
        open=1, high=1, low=1, close=1,
        volume=0.0,
        timeframe="daily",
    )


def _recent_trading_days(n: int) -> list[datetime]:
    """Return the last *n* trading-day datetimes (UTC-naive), oldest first."""
    days, d = [], datetime.utcnow().date()
    while len(days) < n:
        if market_calendar.is_trading_day(d):
            days.append(datetime(d.year, d.month, d.day))
        d -= timedelta(days=1)
    return list(reversed(days))


def _breadth_entry(feeds: list[dict]) -> dict:
    """Extract the breadth/NYAD entry from a check_context_staleness result."""
    matches = [f for f in feeds if f.get("feed_key") == "breadth"]
    assert matches, f"No breadth entry found in feeds: {feeds}"
    return matches[0]


# ─────────────────────────────────────────────────────────────────────────────
# check_context_staleness – NYAD missing
# ─────────────────────────────────────────────────────────────────────────────


class TestNyadMissingFeed:
    """NYAD feed absent while VOO data exists (mirrors the live 404 scenario)."""

    @pytest.mark.asyncio
    async def test_missing_nyad_is_stale(self, db_session):
        """check_context_staleness marks the breadth feed stale when no NYAD rows exist."""
        days = _recent_trading_days(3)
        for ts in days:
            db_session.add(_voo(ts))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        breadth = _breadth_entry(results)

        assert breadth["stale"] is True
        assert breadth[f"latest_breadth"] is None
        assert breadth["latest_voo"] is not None

    @pytest.mark.asyncio
    async def test_missing_nyad_detail_mentions_nyad(self, db_session):
        """The stale detail string references the NYAD feed so operators can act."""
        days = _recent_trading_days(3)
        for ts in days:
            db_session.add(_voo(ts))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        breadth = _breadth_entry(results)

        assert breadth["detail"] is not None
        # Detail must describe what's degraded so operators know the impact.
        detail_lower = breadth["detail"].lower()
        assert "nyad" in detail_lower or "breadth" in detail_lower, (
            f"Expected NYAD/breadth mention in detail: {breadth['detail']}"
        )

    @pytest.mark.asyncio
    async def test_missing_nyad_ticker_field_is_nyad(self, db_session):
        """The ticker field on the breadth entry matches the configured BREADTH_TICKER."""
        days = _recent_trading_days(2)
        for ts in days:
            db_session.add(_voo(ts))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        breadth = _breadth_entry(results)

        assert breadth["ticker"] == settings.BREADTH_TICKER


# ─────────────────────────────────────────────────────────────────────────────
# check_context_staleness – NYAD lagging
# ─────────────────────────────────────────────────────────────────────────────


class TestNyadLaggingFeed:
    """NYAD feed exists but has stopped updating past the allowed lag window."""

    @pytest.mark.asyncio
    async def test_lagging_nyad_exceeds_threshold_is_stale(self, db_session):
        """A breadth feed older than max_lag trading days is marked stale."""
        n = settings.LONG_CONTEXT_STALENESS_MAX_DAYS + 5
        days = _recent_trading_days(n)
        for ts in days:
            db_session.add(_voo(ts))
        # Only the oldest candle has a NYAD entry; everything newer is absent.
        db_session.add(_breadth(days[0]))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        breadth = _breadth_entry(results)

        assert breadth["stale"] is True
        assert breadth["lag_trading_days"] > settings.LONG_CONTEXT_STALENESS_MAX_DAYS

    @pytest.mark.asyncio
    async def test_lagging_nyad_detail_mentions_lag(self, db_session):
        """The detail string for a lagging feed includes the lag count."""
        n = settings.LONG_CONTEXT_STALENESS_MAX_DAYS + 5
        days = _recent_trading_days(n)
        for ts in days:
            db_session.add(_voo(ts))
        db_session.add(_breadth(days[0]))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        breadth = _breadth_entry(results)

        assert breadth["detail"] is not None
        detail_lower = breadth["detail"].lower()
        assert "lag" in detail_lower or "lags" in detail_lower, (
            f"Expected lag mention in detail: {breadth['detail']}"
        )

    @pytest.mark.asyncio
    async def test_nyad_within_threshold_not_stale(self, db_session):
        """A breadth feed lagging by exactly max_lag days is NOT stale."""
        # Need at least max_lag+2 trading days so the cutoff index exists.
        n = settings.LONG_CONTEXT_STALENESS_MAX_DAYS + 3
        days = _recent_trading_days(n)
        for ts in days:
            db_session.add(_voo(ts))
        # Feed present at exactly max_lag days behind the latest VOO candle.
        cutoff_idx = -(settings.LONG_CONTEXT_STALENESS_MAX_DAYS + 1)
        db_session.add(_breadth(days[cutoff_idx]))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        breadth = _breadth_entry(results)

        assert breadth["stale"] is False
        assert breadth["lag_trading_days"] == settings.LONG_CONTEXT_STALENESS_MAX_DAYS

    @pytest.mark.asyncio
    async def test_current_nyad_is_not_stale(self, db_session):
        """A breadth feed current with VOO is not stale and has zero lag."""
        days = _recent_trading_days(5)
        for ts in days:
            db_session.add(_voo(ts))
            db_session.add(_breadth(ts))
        await db_session.flush()

        results = await check_context_staleness(db_session)
        breadth = _breadth_entry(results)

        assert breadth["stale"] is False
        assert breadth["lag_trading_days"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# /api/healthz – NYAD staleness surfaces in context_feeds
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthzNyadStaleness:
    """
    End-to-end: /api/healthz context_feeds block reflects NYAD staleness.

    Calls the healthz() coroutine directly (no HTTP layer) so these tests run
    fast alongside the other healthz unit tests in this directory.
    """

    @pytest.mark.asyncio
    async def test_healthz_context_feeds_includes_breadth(self, db_session):
        """healthz always includes a breadth entry in context_feeds."""
        from routers.predictions import healthz

        body = await healthz(session=db_session)

        assert "context_feeds" in body
        feeds = body["context_feeds"]
        breadth_feeds = [f for f in feeds if f.get("feed_key") == "breadth"]
        assert breadth_feeds, (
            f"Expected a breadth entry in context_feeds; got feed_keys="
            f"{[f.get('feed_key') for f in feeds]}"
        )

    @pytest.mark.asyncio
    async def test_healthz_missing_nyad_marks_status_degraded(self, db_session):
        """
        Missing NYAD while VOO exists → healthz status=degraded.

        This is the live production scenario: Yahoo Finance returns 404 for
        ^NYAD so BreadthCandle stays empty while VOO keeps ingesting.
        """
        from routers.predictions import healthz

        days = _recent_trading_days(3)
        for ts in days:
            db_session.add(_voo(ts))
        await db_session.flush()

        body = await healthz(session=db_session)

        assert body["status"] == "degraded", (
            "Expected status=degraded when NYAD feed is missing"
        )

    @pytest.mark.asyncio
    async def test_healthz_missing_nyad_breadth_entry_is_stale(self, db_session):
        """The breadth entry in context_feeds has stale=True when NYAD is missing."""
        from routers.predictions import healthz

        days = _recent_trading_days(3)
        for ts in days:
            db_session.add(_voo(ts))
        await db_session.flush()

        body = await healthz(session=db_session)

        feeds = body["context_feeds"]
        breadth = next((f for f in feeds if f.get("feed_key") == "breadth"), None)
        assert breadth is not None, "Expected breadth entry in context_feeds"
        assert breadth["stale"] is True, (
            f"Expected breadth stale=True when NYAD is missing; got {breadth}"
        )

    @pytest.mark.asyncio
    async def test_healthz_missing_nyad_emits_context_feed_alert(self, db_session):
        """A missing NYAD feed produces a context_feed alert in the alerts list."""
        from routers.predictions import healthz

        days = _recent_trading_days(3)
        for ts in days:
            db_session.add(_voo(ts))
        await db_session.flush()

        body = await healthz(session=db_session)

        nyad_alerts = [
            a for a in body["alerts"]
            if a.startswith("context_feed") and settings.BREADTH_TICKER in a
        ]
        assert nyad_alerts, (
            f"Expected a context_feed alert for {settings.BREADTH_TICKER}; "
            f"got alerts={body['alerts']}"
        )

    @pytest.mark.asyncio
    async def test_healthz_lagging_nyad_is_stale_in_context_feeds(self, db_session):
        """A NYAD feed that has stopped updating is stale in context_feeds."""
        from routers.predictions import healthz

        n = settings.LONG_CONTEXT_STALENESS_MAX_DAYS + 5
        days = _recent_trading_days(n)
        for ts in days:
            db_session.add(_voo(ts))
        # Only the oldest candle has a breadth row.
        db_session.add(_breadth(days[0]))
        await db_session.flush()

        body = await healthz(session=db_session)

        feeds = body["context_feeds"]
        breadth = next((f for f in feeds if f.get("feed_key") == "breadth"), None)
        assert breadth is not None, "Expected breadth entry in context_feeds"
        assert breadth["stale"] is True, (
            f"Expected breadth stale=True for lagging NYAD; got {breadth}"
        )
        assert body["status"] == "degraded", (
            "Expected status=degraded when NYAD is lagging"
        )

    @pytest.mark.asyncio
    async def test_healthz_current_nyad_not_stale(self, db_session):
        """When NYAD is up-to-date the breadth entry is stale=False and no alert fires."""
        from routers.predictions import healthz

        days = _recent_trading_days(5)
        for ts in days:
            db_session.add(_voo(ts))
            db_session.add(_breadth(ts))
        await db_session.flush()

        body = await healthz(session=db_session)

        feeds = body["context_feeds"]
        breadth = next((f for f in feeds if f.get("feed_key") == "breadth"), None)
        assert breadth is not None, "Expected breadth entry in context_feeds"
        assert breadth["stale"] is False, (
            f"Expected breadth stale=False when NYAD is current; got {breadth}"
        )
        nyad_alerts = [
            a for a in body["alerts"]
            if a.startswith("context_feed") and settings.BREADTH_TICKER in a
        ]
        assert not nyad_alerts, (
            f"Expected no NYAD alert when feed is current; got {nyad_alerts}"
        )
