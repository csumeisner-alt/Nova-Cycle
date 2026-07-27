"""
NovaCycle Data Router
======================
Endpoints for raw candlestick data and indicator snapshots.

GET /api/voo_candles?ticker=VOO&window=30d&timeframe=daily
GET /api/vix_candles?ticker=VOO&window=30d
GET /api/indicators?ticker=VOO
GET /api/gap_status?ticker=VOO
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
