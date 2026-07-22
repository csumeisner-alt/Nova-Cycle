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
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.db import get_db
from database.models import VixCandle, VooCandle
from indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)

router = APIRouter(tags=["data"])

_INDICATORS = TechnicalIndicators()


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

        return {
            "ticker": ticker,
            "timeframe": timeframe,
            "window": window,
            "count": len(candles),
            "candles": candles,
        }
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

        # Also include additional computed metrics
        extras = {
            "return_5d": ind.get("return_5d"),
            "return_10d": ind.get("return_10d"),
            "return_20d": ind.get("return_20d"),
            "sma20_distance": ind.get("sma20_distance"),
        }

        return {
            "ticker": ticker,
            "status": "ok",
            "computed_at": datetime.utcnow().isoformat(),
            "indicators": {**latest, **extras},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_indicators error: %s", exc)
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
                "timestamp": None,
                "session_type": "unknown",
            }

        return {
            "ticker": ticker,
            "gap_percent": latest.gap_percent or 0.0,
            "gap_type": latest.gap_type or "none",
            "timestamp": latest.timestamp.isoformat() if latest.timestamp else None,
            "session_type": latest.session_type,
            "close": latest.close,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_gap_status error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
