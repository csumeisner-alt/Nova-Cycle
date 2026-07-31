"""
NovaCycle Data Router
======================
Endpoints for raw candlestick data and indicator snapshots.

GET /api/voo_candles?ticker=VOO&window=30d&timeframe=daily
GET /api/vix_candles?ticker=VOO&window=30d
GET /api/indicators?ticker=VOO
GET /api/gap_status?ticker=VOO
GET /api/price_snapshot?ticker=VOO
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import asyncio

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.db import get_db
from database.models import VixCandle, VooCandle
from indicators.technical import TechnicalIndicators
from ingestion.fetcher import DataFetcher

logger = logging.getLogger(__name__)

router = APIRouter(tags=["data"])

_INDICATORS = TechnicalIndicators()

# Guards the manual POST /ingest trigger against concurrent runs.
_INGEST_LOCK = asyncio.Lock()

_MARKET_TZ = ZoneInfo("America/New_York")
_LIVE_QUOTE_FETCHER = DataFetcher()


def _trading_day(ts: datetime) -> date:
    """
    Map a stored candle timestamp to its US-market trading day.

    Candle timestamps are stored as UTC-naive datetimes (see DataFetcher /
    market_calendar). Converting to America/New_York before taking the date
    keeps pre-market and regular-session candles on the same trading day even
    if a session crosses midnight UTC.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_MARKET_TZ).date()


def _trading_day_utc_bounds(day: date) -> tuple[datetime, datetime]:
    """Return the UTC-naive [start, end) datetimes covering an ET trading day."""
    start_et = datetime.combine(day, datetime.min.time(), tzinfo=_MARKET_TZ)
    end_et = start_et + timedelta(days=1)
    return (
        start_et.astimezone(timezone.utc).replace(tzinfo=None),
        end_et.astimezone(timezone.utc).replace(tzinfo=None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_ticker(ticker: str) -> str:
    if ticker.upper() != settings.TICKER.upper():
        raise HTTPException(
            status_code=400,
            detail=f"Only '{settings.TICKER}' is supported currently.",
        )
    return ticker.upper()


def _parse_window(window: str) -> timedelta:
    """
    Parse a window string like '7d', '30d', '90d', '1y' into a timedelta.
    Defaults to 30 days on parse error.
    """
    try:
        w = window.strip().lower()
        if w.endswith("y"):
            return timedelta(days=int(w[:-1]) * 365)
        elif w.endswith("d"):
            return timedelta(days=int(w[:-1]))
        elif w.endswith("h"):
            return timedelta(hours=int(w[:-1]))
        return timedelta(days=30)
    except Exception:
        return timedelta(days=30)


def _candle_to_dict(r: VooCandle) -> dict:
    return {
        "id": r.id,
        "ticker": r.ticker,
        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume,
        "timeframe": r.timeframe,
        "is_extended_hours": r.is_extended_hours,
        "session_type": r.session_type,
        "gap_percent": r.gap_percent,
        "gap_type": r.gap_type,
    }


def _valid_ohlc_candle(row: VooCandle) -> bool:
    """Return whether a stored candle has internally consistent OHLC values.

    Ingestion normally removes these rows already, but keeping the endpoint
    defensive prevents a malformed tail row from being shown as a price.
    """
    values = (row.open, row.high, row.low, row.close)
    if any(value is None or float(value) <= 0 for value in values):
        return False
    if row.high < max(row.open, row.close) or row.low > min(row.open, row.close):
        return False
    return True


def _usable_price_candle(row: VooCandle) -> bool:
    """Return whether a candle is a reliable current market-price source."""
    return _valid_ohlc_candle(row) and (
        row.volume is None or float(row.volume) > 0
    )


def _price_point(row: Optional[VooCandle]) -> Optional[dict]:
    if row is None:
        return None
    return {
        "price": float(row.close),
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
    }


async def _latest_usable_candle(
    db: AsyncSession, timeframe: str, *, require_positive_volume: bool = True
) -> Optional[VooCandle]:
    """Find the newest candle matching the requested input-price rules."""
    result = await db.execute(
        select(VooCandle)
        .where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == timeframe,
        )
        .order_by(VooCandle.timestamp.desc())
        .limit(500)
    )
    predicate = _usable_price_candle if require_positive_volume else _valid_ohlc_candle
    return next((row for row in result.scalars().all() if predicate(row)), None)


async def _latest_prior_regular_candle(
    db: AsyncSession, current: Optional[VooCandle]
) -> Optional[VooCandle]:
    """Find the previous regular-session close for the current quote.

    The quote itself may be pre-market, regular-session, or after-hours.  The
    standard day-change comparison is against the last regular-session close
    before the current US-market trading day, so the arrow remains meaningful
    across all three sessions.
    """
    if current is None or current.timestamp is None:
        return None
    try:
        day_start, _ = _trading_day_utc_bounds(_trading_day(current.timestamp))
        result = await db.execute(
            select(VooCandle)
            .where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "5min",
                VooCandle.session_type == "regular",
                VooCandle.timestamp < day_start,
            )
            .order_by(VooCandle.timestamp.desc())
            .limit(500)
        )
        return next(
            (row for row in result.scalars().all() if _usable_price_candle(row)),
            None,
        )
    except Exception as exc:
        logger.warning("Could not find prior regular-session close: %s", exc)
        return None


async def _latest_prior_daily_candle(
    db: AsyncSession, current: Optional[VooCandle]
) -> Optional[VooCandle]:
    """Daily fallback for installations without enough intraday history."""
    current_day = (
        _trading_day(current.timestamp)
        if current is not None and current.timestamp is not None
        else None
    )
    result = await db.execute(
        select(VooCandle)
        .where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == "daily",
        )
        .order_by(VooCandle.timestamp.desc())
        .limit(20)
    )
    rows = [row for row in result.scalars().all() if _usable_price_candle(row)]
    for row in rows:
        if (
            current_day is None
            or row.timestamp is None
            or _trading_day(row.timestamp) < current_day
        ):
            return row
    return None


async def _fetch_live_quote() -> Optional[dict]:
    """Read a fresh vendor quote without making it a model input."""
    try:
        return await asyncio.wait_for(
            _LIVE_QUOTE_FETCHER.fetch_live_quote(),
            timeout=5.0,
        )
    except Exception as exc:
        logger.warning("live_quote_endpoint_fetch_failed error=%s", exc)
        return None


def _vix_to_dict(r: VixCandle) -> dict:
    return {
        "id": r.id,
        "ticker": r.ticker,
        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume,
        "timeframe": r.timeframe,
    }


async def _load_voo_df(db: AsyncSession, limit: int = 500, timeframe: str = "daily") -> pd.DataFrame:
    """Load the most-recent `limit` VOO candles as a DataFrame."""
    result = await db.execute(
        select(VooCandle)
        .where(
            VooCandle.ticker == settings.TICKER,
            VooCandle.timeframe == timeframe,
        )
        .order_by(VooCandle.timestamp.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "timestamp": r.timestamp,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "is_extended_hours": r.is_extended_hours,
            "session_type": r.session_type,
            "gap_percent": r.gap_percent or 0.0,
            "gap_type": r.gap_type,
        }
        for r in rows
    ]
    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


async def _load_vix_df(db: AsyncSession, limit: int = 300) -> pd.DataFrame:
    """Load the most-recent `limit` VIX daily candles as a DataFrame."""
    result = await db.execute(
        select(VixCandle)
        .where(VixCandle.ticker == settings.VIX_TICKER)
        .order_by(VixCandle.timestamp.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "timestamp": r.timestamp,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume or 0.0,
        }
        for r in rows
    ]
    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


async def _compute_gap_momentum(
    db: AsyncSession, as_of: Optional[datetime] = None
) -> Optional[float]:
    """
    Compute gap follow-through momentum at read time (additive; no schema
    changes — nothing is persisted).

    Finds the most recent pre-market 5-min candle carrying a non-zero
    gap_percent, loads that day's regular-session 5-min candles, and applies
    DataFetcher.compute_gap_momentum (price movement over the first 30
    minutes after the open, signed by the gap direction).

    When `as_of` is provided (the timestamp of the latest candle whose gap
    status is being reported), the gap candle must be from the same trading
    day as `as_of` — otherwise None is returned. This prevents pairing
    today's "no gap" status with a stale momentum from an earlier day.

    Returns None when there is no gap candle for that day or not enough
    post-open candles. Never raises.
    """
    try:
        result = await db.execute(
            select(VooCandle)
            .where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "5min",
                VooCandle.session_type == "pre_market",
                VooCandle.gap_percent != 0.0,
            )
            .order_by(VooCandle.timestamp.desc())
            .limit(1)
        )
        gap_candle = result.scalars().first()
        if not gap_candle or not gap_candle.timestamp:
            return None

        # Only report momentum for a gap that belongs to the same US-market
        # trading day (America/New_York) as the candle whose gap status we're
        # returning. Comparing exchange-local dates instead of raw UTC calendar
        # dates keeps pre-market candles paired with that day's regular session
        # even when the session straddles midnight UTC.
        if as_of is not None and _trading_day(gap_candle.timestamp) != _trading_day(as_of):
            return None

        day_start, day_end = _trading_day_utc_bounds(_trading_day(gap_candle.timestamp))
        result = await db.execute(
            select(VooCandle)
            .where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "5min",
                VooCandle.session_type == "regular",
                VooCandle.timestamp >= day_start,
                VooCandle.timestamp < day_end,
            )
            .order_by(VooCandle.timestamp.asc())
            .limit(DataFetcher.GAP_MOMENTUM_CANDLES)
        )
        rows = result.scalars().all()
        if not rows:
            return None
        df = pd.DataFrame(
            [{"timestamp": r.timestamp, "open": r.open, "close": r.close} for r in rows]
        ).set_index("timestamp")
        return DataFetcher.compute_gap_momentum(gap_candle.gap_percent, df)
    except Exception as exc:
        logger.error("_compute_gap_momentum error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/voo_candles")
async def get_voo_candles(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="30d", description="e.g. 7d, 30d, 90d, 1y"),
    timeframe: str = Query(default="daily", description="'daily' or '5min'"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return VOO candlestick data from the database for charting.

    Query parameters:
      ticker:    Always 'VOO' for now
      window:    Time window (e.g. '30d', '90d', '1y')
      timeframe: 'daily' or '5min'
    """
    _validate_ticker(ticker)

    if timeframe not in ("daily", "5min"):
        raise HTTPException(status_code=400, detail="timeframe must be 'daily' or '5min'")

    delta = _parse_window(window)
    since = datetime.utcnow() - delta

    try:
        result = await db.execute(
            select(VooCandle)
            .where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == timeframe,
                VooCandle.timestamp >= since,
            )
            .order_by(VooCandle.timestamp.asc())
        )
        rows = result.scalars().all()
        candles = [_candle_to_dict(r) for r in rows]

        # The Android app expects a flat array of candles, not a wrapped object.
        return candles
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_voo_candles error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/price_snapshot")
async def get_price_snapshot(
    ticker: str = Query(default="VOO"),
    db: AsyncSession = Depends(get_db),
):
    """Return a fresh visible quote and the exact prices used by each model.

    The long model uses the latest valid daily close. The short model uses
    the latest OHLC-valid 5-minute close, including a zero-volume bar because
    the prediction endpoint keeps that bar for feature computation.
    ``current_price`` uses Yahoo's session-specific live quote when available
    (including ``postMarketPrice`` after the regular session). The freshest
    positive-volume candle remains the fallback when the quote vendor is
    unavailable.
    """
    _validate_ticker(ticker)
    try:
        daily = await _latest_usable_candle(db, "daily")
        five_min_model = await _latest_usable_candle(
            db, "5min", require_positive_volume=False
        )
        five_min_current = await _latest_usable_candle(db, "5min")
        current_candle = five_min_current or daily
        live_quote = await _fetch_live_quote()
        current = current_candle
        reference = await _latest_prior_regular_candle(db, current_candle)
        if reference is None:
            reference = await _latest_prior_daily_candle(db, current_candle)
        current_price = (
            live_quote["price"]
            if live_quote is not None
            else (_price_point(current)["price"] if current else None)
        )
        current_timestamp = (
            live_quote["timestamp"]
            if live_quote is not None and live_quote.get("timestamp")
            else (_price_point(current)["timestamp"] if current else None)
        )
        current_session = (
            live_quote["session_type"]
            if live_quote is not None
            else (current.session_type if current else None)
        )
        is_extended_hours = (
            bool(live_quote["is_extended_hours"])
            if live_quote is not None
            else (bool(current.is_extended_hours) if current else False)
        )
        reference_price = _price_point(reference)["price"] if reference else None
        day_change_percent = None
        day_direction = "flat"
        if current_price is not None and reference_price and reference_price > 0:
            day_change_percent = round(
                (current_price - reference_price) / reference_price * 100.0, 4
            )
            if day_change_percent > 0:
                day_direction = "up"
            elif day_change_percent < 0:
                day_direction = "down"
        return {
            "ticker": ticker.upper(),
            "current_price": current_price,
            "current_timestamp": current_timestamp,
            "current_session": current_session,
            "is_extended_hours": is_extended_hours,
            "current_source": (
                live_quote["source"] if live_quote is not None else "stored_candle"
            ),
            "reference_price": reference_price,
            "reference_timestamp": (
                _price_point(reference)["timestamp"] if reference else None
            ),
            "day_change_percent": day_change_percent,
            "day_direction": day_direction,
            "long_model_price": _price_point(daily)["price"] if daily else None,
            "long_model_timestamp": _price_point(daily)["timestamp"] if daily else None,
            "short_model_price": (
                _price_point(five_min_model)["price"] if five_min_model else None
            ),
            "short_model_timestamp": (
                _price_point(five_min_model)["timestamp"] if five_min_model else None
            ),
        }
    except Exception as exc:
        logger.error("get_price_snapshot error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/vix_candles")
async def get_vix_candles(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="30d"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return VIX daily candlestick data from the database.
    """
    _validate_ticker(ticker)

    delta = _parse_window(window)
    since = datetime.utcnow() - delta

    try:
        result = await db.execute(
            select(VixCandle)
            .where(
                VixCandle.ticker == settings.VIX_TICKER,
                VixCandle.timestamp >= since,
            )
            .order_by(VixCandle.timestamp.asc())
        )
        rows = result.scalars().all()
        candles = [_vix_to_dict(r) for r in rows]

        return {
            "ticker": ticker,
            "vix_ticker": settings.VIX_TICKER,
            "window": window,
            "count": len(candles),
            "candles": candles,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_vix_candles error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/indicators")
async def get_indicators(
    ticker: str = Query(default="VOO"),
    db: AsyncSession = Depends(get_db),
):
    """
    Compute and return all current technical indicator values.

    Loads the latest 300 daily candles + 300 VIX candles,
    runs TechnicalIndicators.compute_all(), and returns the 'latest' snapshot.
    """
    _validate_ticker(ticker)

    try:
        voo_df = await _load_voo_df(db, limit=300, timeframe="daily")
        vix_df = await _load_vix_df(db, limit=300)

        if voo_df.empty:
            return {
                "ticker": ticker,
                "status": "no_data",
                "message": "No VOO candles found in database",
                "indicators": {},
            }

        ind = _INDICATORS.compute_all(voo_df, vix_df, exclude_extended=True)
        latest = ind.get("latest", {})

        # Map backend indicator names to the Android app's expected field names.
        # The Android /api/indicators consumer expects a flat object with the
        # indicator values directly at the top level.
        return {
            "ticker": ticker,
            "status": "ok",
            "computed_at": datetime.utcnow().isoformat(),
            "rsi": latest.get("rsi"),
            "stoch_k": latest.get("stoch_k"),
            "stoch_d": latest.get("stoch_d"),
            "stoch_rsi_k": latest.get("stoch_rsi_k"),
            "stoch_rsi_d": latest.get("stoch_rsi_d"),
            "macd_line": latest.get("macd"),
            "macd_signal": latest.get("macd_signal"),
            "macd_histogram": latest.get("macd_histogram"),
            "sma20": latest.get("sma20"),
            "sma50": latest.get("sma50"),
            "sma200": latest.get("sma200"),
            "bollinger_upper": latest.get("bb_upper"),
            "bollinger_lower": latest.get("bb_lower"),
            "bollinger_perc_b": latest.get("bb_pct_b"),
            "cci": latest.get("cci"),
            "williams_r": latest.get("williams_r"),
            "atr": latest.get("atr"),
            "adx": latest.get("adx"),
            "vix_regime": latest.get("vix_regime"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_indicators error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest")
async def trigger_ingest(
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger an incremental data ingestion run (normally scheduled).

    Returns a summary of new candles stored per timeframe. Concurrent calls
    are rejected with 409 so overlapping manual triggers can't hammer the
    upstream data provider.
    """
    # Fail-fast, atomic admission: the locked() check and the acquire happen
    # with no await in between, so on a single event loop no other request can
    # interleave — a concurrent call is rejected with 409 immediately instead
    # of queueing up a back-to-back ingest run.
    if _INGEST_LOCK.locked():
        raise HTTPException(
            status_code=409, detail="An ingestion run is already in progress."
        )
    async with _INGEST_LOCK:
        from main import pipeline  # lazy: avoid circular import at module load

        async def _count(timeframe: str) -> int:
            res = await db.execute(
                select(func.count(VooCandle.id)).where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == timeframe,
                )
            )
            return int(res.scalar() or 0)

        before = {"daily": await _count("daily"), "5min": await _count("5min")}
        started_at = datetime.utcnow()
        try:
            await pipeline.run_incremental_update(db)
            await db.commit()
        except Exception as exc:
            logger.error("manual_ingest_failed error=%s", exc)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

        after = {"daily": await _count("daily"), "5min": await _count("5min")}
        return {
            "status": "ok",
            "triggered": "manual",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "new_candles": {
                "daily": after["daily"] - before["daily"],
                "5min": after["5min"] - before["5min"],
            },
            "total_candles": after,
        }


@router.get("/macro_safety")
async def get_macro_safety(
    ticker: str = Query(default="VOO"),
    db: AsyncSession = Depends(get_db),
):
    """
    Report the current macro safety state in one dedicated endpoint:
      - latest VIX close and its regime (LOW / NORMAL / HIGH / EXTREME)
      - the cached long-trend score and the override thresholds
      - whether the macro override could currently suppress a short signal
      - the most recent signal that actually had the override applied
    """
    _validate_ticker(ticker)

    try:
        from routers import predictions as pred
        from signal_engine.macro_override import (
            LONG_STRONG_BEAR, LONG_STRONG_BULL, ML_OVERRIDE_THRESHOLD,
        )
        from reliability_engine import _classify_vix_regime
        from database.models import SignalHistory

        # Latest VIX close → regime
        res = await db.execute(
            select(VixCandle)
            .where(VixCandle.ticker == settings.VIX_TICKER)
            .order_by(VixCandle.timestamp.desc())
            .limit(1)
        )
        vix = res.scalars().first()
        vix_close = float(vix.close) if vix and vix.close is not None else None
        vix_regime = _classify_vix_regime(vix_close) if vix_close is not None else None

        # ── VIX data-quality flags ────────────────────────────────────────
        # vix_data_missing: no VIX row in the database at all.
        # vix_is_stale: a row exists but it is older than 48 h (covers
        #   weekends; a healthy daily ingest should never exceed ~36 h).
        # vix_staleness_hours: age of the latest VIX row in hours; None
        #   when no row exists.
        vix_data_missing = vix_close is None
        vix_staleness_hours: Optional[float] = None
        vix_is_stale = False
        if vix and vix.timestamp:
            age_hours = (datetime.utcnow() - vix.timestamp).total_seconds() / 3600
            vix_staleness_hours = round(age_hours, 1)
            vix_is_stale = age_hours > 48.0

        long_score = float(pred._last_long_score)
        suppresses_buy = long_score < LONG_STRONG_BEAR
        suppresses_sell = long_score > LONG_STRONG_BULL

        # Most recent signal where the override actually fired
        res = await db.execute(
            select(SignalHistory)
            .where(
                SignalHistory.ticker == settings.TICKER,
                SignalHistory.macro_override_applied == True,  # noqa: E712
            )
            .order_by(SignalHistory.timestamp.desc())
            .limit(1)
        )
        last_override = res.scalars().first()

        if suppresses_buy:
            reason = (
                f"Long trend strongly bearish (score={long_score:.1f} < "
                f"{LONG_STRONG_BEAR}): short BUY signals are suppressed unless "
                f"ML confidence exceeds {ML_OVERRIDE_THRESHOLD:.0%}."
            )
        elif suppresses_sell:
            reason = (
                f"Long trend strongly bullish (score={long_score:.1f} > "
                f"{LONG_STRONG_BULL}): short SELL signals are suppressed unless "
                f"ML confidence exceeds {ML_OVERRIDE_THRESHOLD:.0%}."
            )
        else:
            reason = "No macro override condition is currently active."

        return {
            "ticker": ticker,
            "status": "ok",
            "computed_at": datetime.utcnow().isoformat(),
            "vix_close": vix_close,
            "vix_regime": vix_regime,
            "vix_timestamp": vix.timestamp.isoformat() if vix and vix.timestamp else None,
            "vix_data_missing": vix_data_missing,
            "vix_is_stale": vix_is_stale,
            "vix_staleness_hours": vix_staleness_hours,
            "long_score": long_score,
            "override_active": suppresses_buy or suppresses_sell,
            "suppresses_short_buy": suppresses_buy,
            "suppresses_short_sell": suppresses_sell,
            "reason": reason,
            "thresholds": {
                "long_strong_bear": LONG_STRONG_BEAR,
                "long_strong_bull": LONG_STRONG_BULL,
                "ml_override_threshold": ML_OVERRIDE_THRESHOLD,
            },
            "last_override_applied_at": (
                last_override.timestamp.isoformat()
                if last_override and last_override.timestamp else None
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_macro_safety error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/extended_hours")
async def get_extended_hours(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="7d"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return recent extended-hours session data plus render-ready session
    boundary markers (timestamps where the session type changes), so charts
    can draw pre-market / regular / after-hours separators directly.
    """
    _validate_ticker(ticker)

    delta = _parse_window(window)
    since = datetime.utcnow() - delta

    try:
        result = await db.execute(
            select(VooCandle)
            .where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "5min",
                VooCandle.timestamp >= since,
            )
            .order_by(VooCandle.timestamp.asc())
        )
        rows = result.scalars().all()

        extended = [_candle_to_dict(r) for r in rows if r.is_extended_hours]

        # Session boundary markers: each transition between session types.
        markers = []
        prev_session: Optional[str] = None
        for r in rows:
            session_type = r.session_type or "regular"
            if prev_session is not None and session_type != prev_session:
                markers.append({
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "from_session": prev_session,
                    "to_session": session_type,
                })
            prev_session = session_type

        return {
            "ticker": ticker,
            "window": window,
            "count": len(extended),
            "candles": extended,
            "session_markers": markers,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_extended_hours error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/gap_status")
async def get_gap_status(
    ticker: str = Query(default="VOO"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the gap status from the most recent VOO 5-min candle.

    gap_percent: GapPercent = (PreMarketOpen - PreviousClose) / PreviousClose × 100
    gap_type:    'gap_up' | 'gap_down' | 'none'
    gap_momentum (additive): follow-through of the most recent non-zero gap,
      computed at read time from that day's first 30 minutes of regular-session
      5-min candles (see DataFetcher.compute_gap_momentum). Null when there is
      no gap or not enough post-open candles yet.
    """
    _validate_ticker(ticker)

    try:
        # Get the most recent 5-min candle with gap data
        result = await db.execute(
            select(VooCandle)
            .where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "5min",
            )
            .order_by(VooCandle.timestamp.desc())
            .limit(1)
        )
        latest = result.scalars().first()

        if not latest:
            # Fall back to daily
            result2 = await db.execute(
                select(VooCandle)
                .where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == "daily",
                )
                .order_by(VooCandle.timestamp.desc())
                .limit(1)
            )
            latest = result2.scalars().first()

        if not latest:
            return {
                "ticker": ticker,
                "gap_percent": 0.0,
                "gap_type": "none",
                "gap_class": "none",       # additive
                "gap_momentum": None,      # additive placeholder
                "timestamp": None,
                "session_type": "unknown",
            }

        gap_percent = latest.gap_percent or 0.0
        return {
            "ticker": ticker,
            "gap_percent": gap_percent,
            "gap_type": latest.gap_type or "none",
            # Additive fields (computed at read time; existing fields unchanged)
            "gap_class": DataFetcher.classify_gap_magnitude(gap_percent),
            "gap_momentum": await _compute_gap_momentum(db, as_of=latest.timestamp),
            "timestamp": latest.timestamp.isoformat() if latest.timestamp else None,
            "session_type": latest.session_type,
            "close": latest.close,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_gap_status error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
