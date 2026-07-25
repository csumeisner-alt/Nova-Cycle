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

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func

from database.db import get_session, get_session_factory
from database.models import (
    VooCandle, VixCandle, SpxCandle, ConfidenceHistory, SignalHistory,
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

# ---------------------------------------------------------------------------
# ML fallback tracking: counts how often each endpoint served the neutral 0.5
# fallback instead of a real model prediction, so /api/healthz makes repeated
# fallbacks visible to operators instead of them hiding in logs.
# ---------------------------------------------------------------------------
_ml_fallback_stats: dict = {
    "long_trend": {"count": 0, "last_at": None, "last_reason": None},
    "short_trend": {"count": 0, "last_at": None, "last_reason": None},
}


def _record_ml_fallback(model_name: str, reason: str) -> None:
    """Record that a prediction served the neutral fallback (never raises).

    Increments both the in-memory since-startup counter and the persisted
    cumulative counter (ml/models/ml_fallback_stats.json) so a restart does
    not wipe the evidence of repeated degraded predictions.
    """
    try:
        stats = _ml_fallback_stats[model_name]
        stats["count"] += 1
        stats["last_at"] = datetime.utcnow().isoformat()
        stats["last_reason"] = str(reason)[:300]
        logger.warning("ml_fallback model=%s reason=%s count=%d",
                       model_name, reason, stats["count"])
    except Exception as exc:
        logger.error("_record_ml_fallback error: %s", exc)
    try:
        from ml.fallback_stats import record_fallback
        record_fallback(model_name, reason)
    except Exception as exc:
        logger.error("_record_ml_fallback persist error: %s", exc)

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


async def _load_spx_close_series(session: AsyncSession, limit: int = 300) -> pd.Series:
    """
    Load the daily SPX futures close series for macro sensitivity.

    Returns an empty Series when unavailable so downstream feature code
    keeps the VOO overnight-return fallback (never raises).
    """
    try:
        result = await session.execute(
            select(SpxCandle)
            .where(
                SpxCandle.ticker == settings.SPX_FUTURES_TICKER,
                SpxCandle.timeframe == "daily",
            )
            .order_by(desc(SpxCandle.timestamp))
            .limit(limit)
        )
        rows = result.scalars().all()
        if not rows:
            return pd.Series(dtype=float)
        rows = list(reversed(rows))
        return pd.Series(
            [r.close for r in rows],
            index=pd.to_datetime([r.timestamp for r in rows]),
            dtype=float,
        )
    except Exception as exc:
        logger.error("_load_spx_close_series error: %s", exc)
        return pd.Series(dtype=float)


def _align_spx_to_df(spx_close: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Align the daily SPX close series onto a candle DataFrame's row index
    (the router frames use an integer index with a `timestamp` column).

    Forward-fills the most recent daily SPX close onto each row's date.
    Returns an empty Series on any mismatch so the feature layer keeps
    its VOO overnight fallback (never raises).
    """
    try:
        if spx_close.empty or df.empty or "timestamp" not in df.columns:
            return pd.Series(dtype=float)
        dates = pd.to_datetime(df["timestamp"]).dt.normalize()
        aligned = spx_close.sort_index().reindex(dates, method="ffill")
        aligned.index = df.index
        return aligned.astype(float)
    except Exception as exc:
        logger.error("_align_spx_to_df error: %s", exc)
        return pd.Series(dtype=float)


def _compute_gap_momentum_from_df(df_5min: pd.DataFrame) -> Optional[float]:
    """
    Compute gap follow-through momentum from the already-loaded 5-min frame
    (additive; no extra DB queries, nothing persisted).

    Finds the most recent pre-market candle carrying a non-zero gap_percent,
    takes that day's regular-session candles, and applies
    DataFetcher.compute_gap_momentum (signed % move over the first 30 minutes
    of the regular session relative to the gap direction).

    Returns None (never raises) when there is no gap or not enough
    post-open candles yet.
    """
    from ingestion.fetcher import DataFetcher

    try:
        if df_5min.empty or "gap_percent" not in df_5min.columns:
            return None
        gap_rows = df_5min[
            (df_5min["session_type"] == "pre_market")
            & (df_5min["gap_percent"].fillna(0.0) != 0.0)
        ]
        if gap_rows.empty:
            return None
        gap_row = gap_rows.iloc[-1]
        gap_day = pd.Timestamp(gap_row["timestamp"]).date()
        day_regular = df_5min[
            (df_5min["session_type"] == "regular")
            & (pd.to_datetime(df_5min["timestamp"]).dt.date == gap_day)
        ].sort_values("timestamp")
        if day_regular.empty:
            return None
        post_open = day_regular.head(DataFetcher.GAP_MOMENTUM_CANDLES).set_index("timestamp")
        return DataFetcher.compute_gap_momentum(float(gap_row["gap_percent"]), post_open)
    except Exception as exc:
        logger.error("_compute_gap_momentum_from_df error: %s", exc)
        return None


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

    Each device's stored preferences are checked before firing:
      - Extended-hours signals are skipped when the device opted out.
      - Signals below the device's confidence threshold are silently skipped.

    Uses its own DB session so it can run after the request session is closed.
    Errors are logged but never raise — this must not affect prediction responses.
    """
    from notifications.fcm import FCMNotifier

    try:
        session_factory = get_session_factory()
        async with session_factory() as db:
            result = await db.execute(select(DeviceToken))
            # Load all columns eagerly before the session closes.
            tokens = [
                {
                    "token": t.token,
                    "device_name": t.device_name,
                    "min_buy_threshold": t.min_buy_threshold,
                    "min_sell_threshold": t.min_sell_threshold,
                    "extended_hours_notifications": t.extended_hours_notifications,
                }
                for t in result.scalars().all()
            ]

        if not tokens:
            logger.debug("No device tokens registered — skipping FCM notification")
            return

        notifier = FCMNotifier()
        for device in tokens:
            # ── Preference filtering ────────────────────────────────────────
            # Skip extended-hours signals when the device has opted out.
            if is_extended and not device["extended_hours_notifications"]:
                logger.debug(
                    "Skipping extended-hours notification for device: %s",
                    device["device_name"] or "unknown",
                )
                continue

            # Skip signals below this device's confidence threshold.
            threshold = (
                device["min_buy_threshold"]
                if signal_type == "buy"
                else device["min_sell_threshold"]
            )
            if confidence < threshold:
                logger.debug(
                    "Skipping %s signal (conf=%.2f < threshold=%.2f) for device: %s",
                    signal_type, confidence, threshold,
                    device["device_name"] or "unknown",
                )
                continue
            # ───────────────────────────────────────────────────────────────

            ok = await notifier.send_signal_notification(
                device_token=device["token"],
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
                    "FCM delivery failed for device: %s", device["device_name"] or "unknown"
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
                "ml_fallback": True,
                "liquidity_score": 1.0, "gap_type": "none",
                "macro_override_applied": False,
                "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                "note": "No historical data yet. Run data ingestion first."
            }

        # Compute indicators (exclude extended hours always for long-trend)
        indicators = _indicators_engine.compute_all(daily_df, vix_df, exclude_extended=True)

        # Inject the real SPX futures close series for macro sensitivity
        # (empty series → build_features keeps the VOO overnight fallback).
        spx_aligned = _align_spx_to_df(await _load_spx_close_series(session), daily_df)
        if not spx_aligned.empty:
            indicators["spx_futures_close"] = spx_aligned
        _last_indicators = indicators

        # Build features and run ML model
        ml_fallback = False
        try:
            features = _long_model.build_latest_features(daily_df, indicators)
            if features is None:
                ml_confidence = 0.5
                ml_fallback = True
                _record_ml_fallback("long_trend", "insufficient data for features")
            elif _long_model.is_neutral_fallback():
                # Model missing/stale/failed to load — predict() would silently
                # return 0.5, so flag it explicitly instead.
                ml_confidence = 0.5
                ml_fallback = True
                _record_ml_fallback("long_trend", "model unavailable (missing, stale, or failed to load)")
            else:
                ml_confidence = float(_long_model.predict(features))
                if getattr(_long_model, "last_prediction_was_fallback", False):
                    ml_fallback = True
                    _record_ml_fallback("long_trend", "predict() error fallback (see model logs)")
        except Exception as e:
            ml_confidence = 0.5  # Default to neutral if prediction errors out
            ml_fallback = True
            _record_ml_fallback("long_trend", f"prediction error: {e}")

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
            "ml_fallback": ml_fallback,
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
                "ml_fallback": True,
                "liquidity_score": 1.0, "gap_type": "none",
                "macro_override_applied": False,
                "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                "note": "No 5-min data yet. Run data ingestion first."
            }

        latest = df_5min.iloc[-1]
        is_extended = bool(latest.get("is_extended_hours", False))
        session_type = str(latest.get("session_type", "regular"))
        gap_type = str(latest.get("gap_type", "none"))
        # Gap follow-through momentum (additive; None when no gap / no data yet)
        gap_momentum = _compute_gap_momentum_from_df(df_5min)

        # Compute liquidity score: extended volume / regular volume over last session
        regular_mask = df_5min["session_type"] == "regular"
        extended_mask = df_5min["is_extended_hours"] == True
        regular_vol = float(df_5min.loc[regular_mask, "volume"].sum()) if regular_mask.any() else 1.0
        extended_vol = float(df_5min.loc[extended_mask, "volume"].sum()) if extended_mask.any() else 0.0
        # LiquidityScore = Volume_extended / Volume_regular
        liquidity_score = extended_vol / max(regular_vol, 1.0)

        # Compute short-term indicators from 5-min data
        indicators = _indicators_engine.compute_all(df_5min, vix_df, exclude_extended=False)

        # Inject the real SPX futures close series for macro sensitivity
        # (empty series → build_features keeps the VOO overnight fallback).
        spx_aligned = _align_spx_to_df(await _load_spx_close_series(session), df_5min)
        if not spx_aligned.empty:
            indicators["spx_futures_close"] = spx_aligned

        # Build features and predict
        ml_fallback = False
        try:
            features = _short_model.build_latest_features(df_5min, indicators)
            if features is None:
                ml_confidence = 0.5
                ml_fallback = True
                _record_ml_fallback("short_trend", "insufficient data for features")
            elif _short_model.is_neutral_fallback():
                # Model missing/stale/failed to load — predict() would silently
                # return 0.5, so flag it explicitly instead.
                ml_confidence = 0.5
                ml_fallback = True
                _record_ml_fallback("short_trend", "model unavailable (missing, stale, or failed to load)")
            else:
                ml_confidence = float(_short_model.predict(features))
                if getattr(_short_model, "last_prediction_was_fallback", False):
                    ml_fallback = True
                    _record_ml_fallback("short_trend", "predict() error fallback (see model logs)")
        except Exception as e:
            ml_confidence = 0.5
            ml_fallback = True
            _record_ml_fallback("short_trend", f"prediction error: {e}")

        # Compute short gauge score
        # age_in_minutes=0 = latest candle gets full weight
        result = _short_gauge.compute_score(
            indicators, ml_confidence,
            is_extended=is_extended,
            liquidity_score=liquidity_score,
            gap_type=gap_type,
            age_in_minutes=0,
            gap_momentum=gap_momentum
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
            "ml_fallback": ml_fallback,
            "liquidity_score": liquidity_score,
            "gap_type": gap_type,
            "gap_momentum": gap_momentum,
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
        # NOTE: _last_indicators["vix_regime"] is a per-row pandas Series;
        # the scalar regime lives in the "latest" sub-dict. Passing the Series
        # here used to crash (.upper() on a Series) and 500 the endpoint.
        vix_regime = _last_indicators.get("latest", {}).get("vix_regime") or "NORMAL"
        if not isinstance(vix_regime, str):
            vix_regime = "NORMAL"
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
async def healthz(session: AsyncSession = Depends(get_session)):
    """
    Health check endpoint.

    Also reports ML model health so a failed weekly retrain (or a model
    stuck serving the neutral 0.5 fallback) is visible, not just logged.
    Overall status becomes "degraded" if either model failed its last
    training attempt or is running in neutral-fallback mode.
    """
    from ml.training_status import (
        get_training_status,
        CONSECUTIVE_FAILURE_ALERT_THRESHOLD,
    )
    from ml.fallback_stats import get_persisted_fallback_stats, get_last_reset_at
    from database.models import ModelMetadata

    training_status = get_training_status()
    persisted_fallbacks = get_persisted_fallback_stats()
    try:
        fallback_last_reset_at = get_last_reset_at()
    except Exception as exc:
        logger.error("healthz: fallback last-reset lookup failed: %s", exc)
        fallback_last_reset_at = None

    models = {}
    degraded = False
    for name, model in (("long_trend", _long_model), ("short_trend", _short_model)):
        neutral = True
        try:
            neutral = model.is_neutral_fallback()
        except Exception as exc:
            logger.error("healthz: %s fallback check failed: %s", name, exc)

        last_trained_at = None
        try:
            result = await session.execute(
                select(func.max(ModelMetadata.trained_at)).where(
                    ModelMetadata.model_name == name
                )
            )
            ts = result.scalar()
            last_trained_at = ts.isoformat() if ts else None
        except Exception as exc:
            logger.error("healthz: %s metadata lookup failed: %s", name, exc)

        status = training_status.get(name, {})
        failed = status.get("success") is False
        consecutive_failures = status.get("consecutive_failures") or 0
        training_stuck = consecutive_failures >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD
        fallback_stats = _ml_fallback_stats.get(name, {})
        persisted = persisted_fallbacks.get(name, {})
        if failed or neutral or fallback_stats.get("count", 0) > 0:
            degraded = True

        models[name] = {
            "last_training_success": status.get("success"),
            "consecutive_training_failures": consecutive_failures,
            "training_stuck": training_stuck,
            "last_training_error": status.get("error"),
            "last_training_attempted_at": status.get("attempted_at"),
            "last_training_accuracy": status.get("accuracy"),
            "last_trained_at": last_trained_at,
            "neutral_fallback": neutral,
            "ml_fallback_count": fallback_stats.get("count", 0),
            "ml_fallback_last_at": fallback_stats.get("last_at"),
            "ml_fallback_last_reason": fallback_stats.get("last_reason"),
            "ml_fallback_total_count": persisted.get("total_count", 0),
            "ml_fallback_total_last_at": persisted.get("last_at"),
            "ml_fallback_total_last_reason": persisted.get("last_reason"),
        }

    # ── SPX futures staleness ────────────────────────────────────────────
    from ingestion.pipeline import (
        check_spx_staleness, check_vix_staleness, check_5min_staleness,
        get_5min_recovery_status,
    )

    spx_data = None
    try:
        spx_data = await check_spx_staleness(session)
        if spx_data.get("stale"):
            degraded = True
    except Exception as exc:
        logger.error("healthz: SPX staleness check failed: %s", exc)

    # ── VIX staleness ─────────────────────────────────────────────────────
    vix_data = None
    try:
        vix_data = await check_vix_staleness(session)
        if vix_data.get("stale"):
            degraded = True
    except Exception as exc:
        logger.error("healthz: VIX staleness check failed: %s", exc)

    # ── VOO 5-min staleness ───────────────────────────────────────────────
    fivemin_data = None
    try:
        fivemin_data = await check_5min_staleness(session)
        if fivemin_data.get("stale"):
            degraded = True
    except Exception as exc:
        logger.error("healthz: 5-min staleness check failed: %s", exc)

    # ── VOO 5-min stall auto-recovery (last attempt, if any) ─────────────
    fivemin_recovery = None
    try:
        fivemin_recovery = get_5min_recovery_status()
    except Exception as exc:
        logger.error("healthz: 5-min recovery status lookup failed: %s", exc)

    alerts = []
    if spx_data and spx_data.get("stale"):
        alerts.append(f"spx_futures: {spx_data.get('detail')}")
    if vix_data and vix_data.get("stale"):
        alerts.append(f"vix: {vix_data.get('detail')}")
    if fivemin_data and fivemin_data.get("stale"):
        alerts.append(f"voo_5min: {fivemin_data.get('detail')}")
    if fivemin_recovery and fivemin_recovery.get("outcome") == "failed":
        alerts.append(
            "voo_5min_recovery: last auto-recovery attempt failed "
            f"(at {fivemin_recovery.get('last_attempt_at')}, "
            f"bars_fetched={fivemin_recovery.get('bars_fetched')})"
        )
    for name, info in models.items():
        if info["last_training_success"] is False:
            alerts.append(
                f"{name}: last training attempt failed"
                + (f" ({info['last_training_error']})" if info["last_training_error"] else "")
            )
        if info["training_stuck"]:
            alerts.append(
                f"{name}: training stuck — {info['consecutive_training_failures']} "
                f"consecutive failed retrain attempts (threshold "
                f"{CONSECUTIVE_FAILURE_ALERT_THRESHOLD}); model is running on an "
                "increasingly stale last-good version"
            )
        if info["neutral_fallback"]:
            alerts.append(f"{name}: model unavailable — serving neutral 0.5 predictions")
        if info["ml_fallback_count"] > 0:
            alerts.append(
                f"{name}: served neutral-fallback predictions {info['ml_fallback_count']} time(s) "
                f"since startup (last: {info['ml_fallback_last_reason']} at {info['ml_fallback_last_at']})"
            )

    return {
        "status": "degraded" if degraded else "ok",
        "ticker": "VOO",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "NovaCycle API",
        "models": models,
        "spx_futures": spx_data,
        "vix": vix_data,
        "voo_5min": fivemin_data,
        "voo_5min_recovery": fivemin_recovery,
        "alerts": alerts,
        "fallback_stats_last_reset_at": fallback_last_reset_at,
        "note": "Pipeline currently fetches only VOO. Multi-ticker ingestion will be added later."
    }


# ---------------------------------------------------------------------------
# POST /admin/reset_fallback_stats  (operator-only)
# ---------------------------------------------------------------------------
def _require_admin_token(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """Guard for operator-only endpoints.

    Compares the X-Admin-Token header against ADMIN_TOKEN (falling back to
    SESSION_SECRET). If neither is configured the endpoint is disabled (503)
    instead of being left open.
    """
    import secrets as _secrets
    expected = settings.ADMIN_TOKEN or settings.SESSION_SECRET
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin endpoints disabled: set ADMIN_TOKEN (or SESSION_SECRET)",
        )
    if not x_admin_token or not _secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=403, detail="Invalid or missing X-Admin-Token")


@router.post("/admin/reset_fallback_stats", dependencies=[Depends(_require_admin_token)])
async def reset_fallback_stats_endpoint():
    """
    Operator action: clear the persisted cumulative ML-fallback history
    (ml/models/ml_fallback_stats.json) *and* the in-memory since-startup
    counters, after the root cause of the fallbacks has been fixed.

    /api/healthz reflects the reset immediately: ml_fallback_count and
    ml_fallback_total_count drop to 0 and their alerts disappear (unless a
    model is still actively degraded, which will re-increment the counters).
    """
    from ml.fallback_stats import reset_fallback_stats

    try:
        previous = reset_fallback_stats()
    except Exception as exc:
        logger.error("reset_fallback_stats failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to reset fallback stats: {exc}")

    # Also clear in-memory since-startup counters so healthz is clean now,
    # not after the next restart.
    previous_in_memory = {}
    for name, stats in _ml_fallback_stats.items():
        previous_in_memory[name] = dict(stats)
        stats["count"] = 0
        stats["last_at"] = None
        stats["last_reason"] = None

    logger.info("ML fallback history reset by operator")
    return {
        "status": "ok",
        "reset_at": datetime.now(timezone.utc).isoformat(),
        "previous_persisted": previous,
        "previous_in_memory": previous_in_memory,
    }
