"""
NovaCycle - Predictions Router
Handles all ML prediction and signal endpoints.
NOTE: Multi-ticker support placeholder - only 'VOO' is accepted.
"""

import asyncio
import logging
import uuid
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_

from database.db import get_session, get_session_factory
from database.models import (
    VooCandle, VixCandle, ConfidenceHistory, SignalHistory,
    TradeCycles, FilteredSignal, DeviceToken
)
from indicators.technical import TechnicalIndicators
from ml.long_trend import LongTrendModel
from ml.short_trend import ShortTrendModel
from ml.hold_time import HoldTimePredictionEngine
from signal_engine.long_gauge import LongTrendGauge
from signal_engine.short_gauge import ShortTrendGauge
from signal_engine.macro_override import MacroOverrideSafety
from config import settings

import pandas as pd
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache for last computed scores (used by hold_time, macro override)
# ---------------------------------------------------------------------------
_last_long_score: float = 0.0
_last_short_score: float = 0.0
_last_indicators: dict = {}

# Singleton instances
_indicators_engine = TechnicalIndicators()
_long_gauge = LongTrendGauge()
_short_gauge = ShortTrendGauge()
_macro_override = MacroOverrideSafety()
_long_model = LongTrendModel()
_short_model = ShortTrendModel()
_hold_engine = HoldTimePredictionEngine()


def _validate_ticker(ticker: str):
    """Validate ticker is VOO. Multi-ticker support will be added later."""
    if ticker.upper() != "VOO":
        raise HTTPException(
            status_code=400,
            detail=f"Ticker '{ticker}' not supported. Only 'VOO' is accepted in this version."
        )


def _parse_window(window: str) -> timedelta:
    """Parse window string like '7d', '30d', '24h' into timedelta."""
    if window.endswith("d"):
        return timedelta(days=int(window[:-1]))
    elif window.endswith("h"):
        return timedelta(hours=int(window[:-1]))
    else:
        return timedelta(days=30)


async def _load_daily_candles(session: AsyncSession, ticker: str, limit: int = 300) -> pd.DataFrame:
    """Load regular-hours daily candles from DB (extended hours excluded)."""
    result = await session.execute(
        select(VooCandle)
        .where(and_(
            VooCandle.ticker == ticker,
            VooCandle.timeframe == "daily",
            VooCandle.is_extended_hours == False
        ))
        .order_by(desc(VooCandle.timestamp))
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()
    rows = list(reversed(rows))
    return pd.DataFrame([{
        "timestamp": r.timestamp,
        "open": r.open, "high": r.high, "low": r.low, "close": r.close,
        "volume": r.volume, "is_extended_hours": r.is_extended_hours,
        "session_type": r.session_type, "gap_percent": r.gap_percent,
        "gap_type": r.gap_type, "ticker": r.ticker
    } for r in rows])


async def _load_5min_candles(session: AsyncSession, ticker: str, limit: int = 500) -> pd.DataFrame:
    """Load 5-minute candles from DB (all sessions)."""
    result = await session.execute(
        select(VooCandle)
        .where(and_(VooCandle.ticker == ticker, VooCandle.timeframe == "5min"))
        .order_by(desc(VooCandle.timestamp))
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()
    rows = list(reversed(rows))
    return pd.DataFrame([{
        "timestamp": r.timestamp,
        "open": r.open, "high": r.high, "low": r.low, "close": r.close,
        "volume": r.volume, "is_extended_hours": r.is_extended_hours,
        "session_type": r.session_type, "gap_percent": r.gap_percent,
        "gap_type": r.gap_type, "ticker": r.ticker
    } for r in rows])


async def _load_vix_candles(session: AsyncSession, limit: int = 300) -> pd.DataFrame:
    """Load VIX daily candles."""
    result = await session.execute(
        select(VixCandle)
        .order_by(desc(VixCandle.timestamp))
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()
    rows = list(reversed(rows))
    return pd.DataFrame([{
        "timestamp": r.timestamp,
        "open": r.open, "high": r.high, "low": r.low, "close": r.close,
        "ticker": r.ticker
    } for r in rows])


async def _store_confidence(session: AsyncSession, ticker: str, long_buy: float,
                            long_sell: float, short_buy: float, short_sell: float,
                            session_type: str, is_extended: bool):
    """Persist confidence snapshot to confidence_history table."""
    entry = ConfidenceHistory(
        timestamp=datetime.utcnow(),
        ticker=ticker,
        long_buy_confidence=long_buy,
        long_sell_confidence=long_sell,
        short_buy_confidence=short_buy,
        short_sell_confidence=short_sell,
        session_type=session_type,
        is_extended_hours=is_extended
    )
    session.add(entry)
    await session.commit()


async def _store_signal(session: AsyncSession, ticker: str, signal_type: str,
                        gauge_type: str, confidence: float, session_type: str,
                        is_extended: bool, gap_type: str, liquidity_score: float,
                        macro_override: bool, cycle_id: Optional[str] = None):
    """Persist signal event to signal_history table."""
    entry = SignalHistory(
        timestamp=datetime.utcnow(),
        ticker=ticker,
        cycle_id=cycle_id or str(uuid.uuid4()),
        signal_type=signal_type,
        gauge_type=gauge_type,
        confidence=confidence,
        session_type=session_type,
        is_extended_hours=is_extended,
        gap_type=gap_type,
        liquidity_score=liquidity_score,
        macro_override_applied=macro_override
    )
    session.add(entry)
    await session.commit()
    return entry


async def _notify_all_devices_bg(
    signal_type: str,
    gauge_type: str,
    confidence: float,
    is_extended: bool,
    gap_type: str,
    liquidity_score: float,
    score: float,
) -> None:
    """
    Background task: send FCM push notifications to all registered device tokens.

    Uses its own DB session so it can run after the request session is closed.
    Errors are logged but never raise — this must not affect prediction responses.
    """
    from notifications.fcm import FCMNotifier

    try:
        session_factory = get_session_factory()
        async with session_factory() as db:
            result = await db.execute(select(DeviceToken))
            tokens = result.scalars().all()

        if not tokens:
            logger.debug("No device tokens registered — skipping FCM notification")
            return

        notifier = FCMNotifier()
        for device in tokens:
            ok = await notifier.send_signal_notification(
                device_token=device.token,
                signal_type=signal_type,
                gauge_type=gauge_type,
                confidence=confidence,
                is_extended=is_extended,
                score=score,
                gap_type=gap_type,
                liquidity_score=liquidity_score,
            )
            if not ok:
                logger.warning(
                    "FCM delivery failed for device: %s", device.device_name or "unknown"
                )
    except Exception as exc:
        logger.error("Background FCM notification error: %s", exc)


# ---------------------------------------------------------------------------
# POST /predict_long
# ---------------------------------------------------------------------------
@router.post("/predict_long")
async def predict_long(
    ticker: str = Query(default="VOO"),
    session: AsyncSession = Depends(get_session)
):
    """
    Run long-trend prediction using daily candles + long-trend ML model.
    Extended-hours data is never used for long-trend predictions.
    """
    global _last_long_score, _last_indicators
    _validate_ticker(ticker)

    try:
        daily_df = await _load_daily_candles(session, ticker, limit=300)
        vix_df = await _load_vix_candles(session, limit=300)

        if daily_df.empty:
            # Return neutral if no data yet
            return {
                "score": 0, "signal": "neutral", "confidence": 0.5,
                "indicator_breakdown": {}, "ml_confidence": 0.5,
                "liquidity_score": 1.0, "gap_type": "none",
                "macro_override_applied": False,
                "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                "note": "No historical data yet. Run data ingestion first."
            }

        # Compute indicators (exclude extended hours always for long-trend)
        indicators = _indicators_engine.compute_all(daily_df, vix_df, exclude_extended=True)
        _last_indicators = indicators

        # Build features and run ML model
        try:
            features = _long_model.build_features(daily_df, indicators)
            ml_confidence = float(_long_model.predict(features))
        except Exception as e:
            ml_confidence = 0.5  # Default to neutral if model not trained

        # Compute gauge score (age_in_days=0 = latest candle, full weight)
        result = _long_gauge.compute_score(indicators, ml_confidence, age_in_days=0)
        _last_long_score = result["score"]

        # Determine session info from latest candle
        latest = daily_df.iloc[-1]
        session_type = str(latest.get("session_type", "regular"))
        is_extended = bool(latest.get("is_extended_hours", False))

        # Map confidence direction
        long_buy_conf = ml_confidence if result["signal"] == "buy" else (1 - ml_confidence)
        long_sell_conf = ml_confidence if result["signal"] == "sell" else (1 - ml_confidence)

        # Persist confidence history
        await _store_confidence(session, ticker,
                                long_buy=long_buy_conf, long_sell=long_sell_conf,
                                short_buy=0.0, short_sell=0.0,
                                session_type=session_type, is_extended=is_extended)

        # Persist signal if actionable, then push notification in background
        if result["signal"] in ("buy", "sell"):
            await _store_signal(
                session, ticker,
                signal_type=result["signal"], gauge_type="long",
                confidence=abs(result["score"]) / 100.0,
                session_type=session_type, is_extended=is_extended,
                gap_type=str(latest.get("gap_type", "none")),
                liquidity_score=1.0, macro_override=False
            )
            asyncio.create_task(_notify_all_devices_bg(
                signal_type=result["signal"],
                gauge_type="long",
                confidence=abs(result["score"]) / 100.0,
                is_extended=is_extended,
                gap_type=str(latest.get("gap_type", "none")),
                liquidity_score=1.0,
                score=result["score"],
            ))

        return {
            "score": result["score"],
            "signal": result["signal"],
            "confidence": result["confidence"],
            "indicator_breakdown": result.get("breakdown", {}),
            "ml_confidence": ml_confidence,
            "liquidity_score": 1.0,
            "gap_type": str(latest.get("gap_type", "none")),
            "macro_override_applied": False,
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Long prediction error: {str(e)}")


# ---------------------------------------------------------------------------
# POST /predict_short
# ---------------------------------------------------------------------------
@router.post("/predict_short")
async def predict_short(
    ticker: str = Query(default="VOO"),
    session: AsyncSession = Depends(get_session)
):
    """
    Run short-trend prediction using 5-min candles + short-trend ML model.
    Extended-hours candles are included but weighted at 0.5.
    Macro override safety layer is always applied.
    """
    global _last_short_score
    _validate_ticker(ticker)

    try:
        df_5min = await _load_5min_candles(session, ticker, limit=500)
        vix_df = await _load_vix_candles(session, limit=50)

        if df_5min.empty:
            return {
                "score": 0, "signal": "neutral", "confidence": 0.5,
                "indicator_breakdown": {}, "ml_confidence": 0.5,
                "liquidity_score": 1.0, "gap_type": "none",
                "macro_override_applied": False,
                "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                "note": "No 5-min data yet. Run data ingestion first."
            }

        latest = df_5min.iloc[-1]
        is_extended = bool(latest.get("is_extended_hours", False))
        session_type = str(latest.get("session_type", "regular"))
        gap_type = str(latest.get("gap_type", "none"))

        # Compute liquidity score: extended volume / regular volume over last session
        regular_mask = df_5min["session_type"] == "regular"
        extended_mask = df_5min["is_extended_hours"] == True
        regular_vol = float(df_5min.loc[regular_mask, "volume"].sum()) if regular_mask.any() else 1.0
        extended_vol = float(df_5min.loc[extended_mask, "volume"].sum()) if extended_mask.any() else 0.0
        # LiquidityScore = Volume_extended / Volume_regular
        liquidity_score = extended_vol / max(regular_vol, 1.0)

        # Compute short-term indicators from 5-min data
        indicators = _indicators_engine.compute_all(df_5min, vix_df, exclude_extended=False)

        # Build features and predict
        try:
            features = _short_model.build_features(df_5min, indicators)
            ml_confidence = float(_short_model.predict(features))
        except Exception:
            ml_confidence = 0.5

        # Compute short gauge score
        # age_in_minutes=0 = latest candle gets full weight
        result = _short_gauge.compute_score(
            indicators, ml_confidence,
            is_extended=is_extended,
            liquidity_score=liquidity_score,
            gap_type=gap_type,
            age_in_minutes=0
        )
        _last_short_score = result["score"]

        # Apply macro override safety layer
        override_result = _macro_override.apply_override(
            long_score=_last_long_score,
            short_signal=result["signal"],
            short_ml_confidence=ml_confidence
        )
        final_signal = result["signal"]
        macro_override_applied = override_result["override_applied"]
        if override_result["override_applied"]:
            final_signal = "neutral"

        # Confidence values
        short_buy_conf = ml_confidence if final_signal == "buy" else max(0.0, ml_confidence - 0.3)
        short_sell_conf = ml_confidence if final_signal == "sell" else max(0.0, ml_confidence - 0.3)

        # Persist confidence history
        await _store_confidence(session, ticker,
                                long_buy=0.0, long_sell=0.0,
                                short_buy=short_buy_conf, short_sell=short_sell_conf,
                                session_type=session_type, is_extended=is_extended)

        # Persist signal if actionable, then push notification in background
        if final_signal in ("buy", "sell"):
            await _store_signal(
                session, ticker,
                signal_type=final_signal, gauge_type="short",
                confidence=abs(result["score"]) / 100.0,
                session_type=session_type, is_extended=is_extended,
                gap_type=gap_type, liquidity_score=liquidity_score,
                macro_override=macro_override_applied
            )
            asyncio.create_task(_notify_all_devices_bg(
                signal_type=final_signal,
                gauge_type="short",
                confidence=abs(result["score"]) / 100.0,
                is_extended=is_extended,
                gap_type=gap_type,
                liquidity_score=liquidity_score,
                score=result["score"],
            ))

        return {
            "score": result["score"],
            "signal": final_signal,
            "confidence": result["confidence"],
            "indicator_breakdown": result.get("breakdown", {}),
            "ml_confidence": ml_confidence,
            "liquidity_score": liquidity_score,
            "gap_type": gap_type,
            "macro_override_applied": macro_override_applied,
            "macro_override_reason": override_result.get("reason", ""),
            "session_type": session_type,
            "is_extended_hours": is_extended,
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Short prediction error: {str(e)}")


# ---------------------------------------------------------------------------
# POST /hold_time_estimate
# ---------------------------------------------------------------------------
@router.post("/hold_time_estimate")
async def hold_time_estimate(
    ticker: str = Query(default="VOO"),
    session: AsyncSession = Depends(get_session)
):
    """Estimate expected hold time based on current market conditions."""
    _validate_ticker(ticker)
    try:
        vix_regime = _last_indicators.get("vix_regime", "NORMAL")
        result = _hold_engine.estimate_hold_time(
            indicators=_last_indicators,
            long_score=_last_long_score,
            short_score=_last_short_score,
            vix_regime=vix_regime
        )
        result["ticker"] = ticker
        result["timestamp"] = datetime.utcnow().isoformat()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hold time error: {str(e)}")


# ---------------------------------------------------------------------------
# GET /confidence_history
# ---------------------------------------------------------------------------
@router.get("/confidence_history")
async def confidence_history(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="7d"),
    session: AsyncSession = Depends(get_session)
):
    """Return confidence history for the given ticker and time window."""
    _validate_ticker(ticker)
    since = datetime.utcnow() - _parse_window(window)
    result = await session.execute(
        select(ConfidenceHistory)
        .where(and_(
            ConfidenceHistory.ticker == ticker,
            ConfidenceHistory.timestamp >= since
        ))
        .order_by(desc(ConfidenceHistory.timestamp))
        .limit(1000)
    )
    rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "ticker": r.ticker,
            "long_buy_confidence": r.long_buy_confidence,
            "long_sell_confidence": r.long_sell_confidence,
            "short_buy_confidence": r.short_buy_confidence,
            "short_sell_confidence": r.short_sell_confidence,
            "session_type": r.session_type,
            "is_extended_hours": r.is_extended_hours
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# GET /signal_history
# ---------------------------------------------------------------------------
@router.get("/signal_history")
async def signal_history(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="30d"),
    session: AsyncSession = Depends(get_session)
):
    """Return all raw BUY/SELL signals for the given window."""
    _validate_ticker(ticker)
    since = datetime.utcnow() - _parse_window(window)
    result = await session.execute(
        select(SignalHistory)
        .where(and_(
            SignalHistory.ticker == ticker,
            SignalHistory.timestamp >= since
        ))
        .order_by(SignalHistory.timestamp)
    )
    rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "ticker": r.ticker,
            "cycle_id": r.cycle_id,
            "signal_type": r.signal_type,
            "gauge_type": r.gauge_type,
            "confidence": r.confidence,
            "session_type": r.session_type,
            "is_extended_hours": r.is_extended_hours,
            "gap_type": r.gap_type,
            "liquidity_score": r.liquidity_score,
            "macro_override_applied": r.macro_override_applied
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# GET /filtered_signal_history  (strongest-confidence rule)
# ---------------------------------------------------------------------------
@router.get("/filtered_signal_history")
async def filtered_signal_history(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="30d"),
    session: AsyncSession = Depends(get_session)
):
    """
    Apply strongest-confidence rule to produce alternating BUY→SELL signals:
    1. Load all signals in window, sorted by timestamp
    2. Group consecutive signals of the same type
    3. Keep the highest-confidence signal per group
    4. Enforce strict alternation: BUY → SELL → BUY → ...
    5. Assign cycle_id to each BUY→SELL pair
    """
    _validate_ticker(ticker)
    since = datetime.utcnow() - _parse_window(window)
    result = await session.execute(
        select(SignalHistory)
        .where(and_(
            SignalHistory.ticker == ticker,
            SignalHistory.timestamp >= since,
            SignalHistory.signal_type.in_(["buy", "sell"])
        ))
        .order_by(SignalHistory.timestamp)
    )
    rows = result.scalars().all()

    if not rows:
        return []

    # Step 1: Group consecutive same-type signals
    groups = []
    current_group = [rows[0]]
    for row in rows[1:]:
        if row.signal_type == current_group[-1].signal_type:
            current_group.append(row)
        else:
            groups.append(current_group)
            current_group = [row]
    groups.append(current_group)

    # Step 2: Select highest-confidence signal from each group
    best_signals = []
    for group in groups:
        best = max(group, key=lambda r: r.confidence)
        best_signals.append(best)

    # Step 3: Enforce strict alternation BUY→SELL→BUY
    filtered = []
    last_type = None
    for sig in best_signals:
        if sig.signal_type != last_type:
            filtered.append(sig)
            last_type = sig.signal_type

    # Step 4: Assign cycle_id to BUY→SELL pairs
    result_list = []
    pending_buy = None
    for sig in filtered:
        if sig.signal_type == "buy":
            pending_buy = sig
            result_list.append({
                "id": sig.id,
                "timestamp": sig.timestamp.isoformat(),
                "ticker": sig.ticker,
                "signal_type": sig.signal_type,
                "gauge_type": sig.gauge_type,
                "confidence": sig.confidence,
                "cycle_id": None,  # assigned when SELL found
                "session_type": sig.session_type,
                "is_extended_hours": sig.is_extended_hours
            })
        elif sig.signal_type == "sell" and pending_buy is not None:
            cycle_id = str(uuid.uuid4())
            # Update the buy entry's cycle_id
            if result_list:
                result_list[-1]["cycle_id"] = cycle_id
            result_list.append({
                "id": sig.id,
                "timestamp": sig.timestamp.isoformat(),
                "ticker": sig.ticker,
                "signal_type": sig.signal_type,
                "gauge_type": sig.gauge_type,
                "confidence": sig.confidence,
                "cycle_id": cycle_id,
                "session_type": sig.session_type,
                "is_extended_hours": sig.is_extended_hours
            })
            pending_buy = None

    return result_list


# ---------------------------------------------------------------------------
# GET /trade_history  (Signal Reliability Metrics)
# ---------------------------------------------------------------------------
@router.get("/trade_history")
async def trade_history(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="30d"),
    session: AsyncSession = Depends(get_session)
):
    """
    Return BUY→SELL trade cycles and their reliability summary for the ticker.

    This endpoint delegates all reliability logic to reliability_engine.py so that
    existing signal filtering and prediction code is not touched. It regenerates
    cycles from the filtered signal timeline, persists any new ones, and returns
    both the cycle list and aggregate metrics.
    """
    _validate_ticker(ticker)
    from reliability_engine import get_trade_history_with_metrics
    return await get_trade_history_with_metrics(session, ticker=ticker, window=window)


# ---------------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------------
@router.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {
        "status": "ok",
        "ticker": "VOO",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "NovaCycle API",
        "note": "Pipeline currently fetches only VOO. Multi-ticker ingestion will be added later."
    }
