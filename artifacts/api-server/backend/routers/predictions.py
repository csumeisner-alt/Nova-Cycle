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
from typing import List, Optional

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
from signal_engine.decision_filter import DecisionFilter
from signal_engine.conviction import ConvictionEvaluator, TIER_HIGH_CONVICTION
from signal_engine.normalization import (
    normalize_gauge_output, reconcile_display_signal, NEUTRAL_DEFAULTS,
)
from config import settings
from ingestion.fetcher import ohlc_validation_issue

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

# In-memory cache for last computed buy confidences (used by decision-filter
# divergence checks). These are updated by predict_long / predict_short and
# are safe because the backend runs as a single-process Reserved VM.
_last_long_buy_conf: float = 0.5
_last_short_buy_conf: float = 0.5
_last_long_sell_conf: float = 0.5
_last_short_sell_conf: float = 0.5

# ---------------------------------------------------------------------------
# ML fallback tracking: counts how often each endpoint served the neutral 0.5
# fallback instead of a real model prediction, so /api/healthz makes repeated
# fallbacks visible to operators instead of them hiding in logs.
# ---------------------------------------------------------------------------
_ml_fallback_stats: dict = {
    "long_trend": {"count": 0, "last_at": None, "last_reason": None},
    "short_trend": {"count": 0, "last_at": None, "last_reason": None},
}

# ---------------------------------------------------------------------------
# OHLC quarantine tracking: records when a malformed candle was detected at
# prediction time so /api/healthz can surface the condition to operators.
# ---------------------------------------------------------------------------
_ohlc_quarantine_stats: dict = {
    "count": 0,        # total bad candles seen since startup
    "last_at": None,   # ISO timestamp of the most recent detection
    "last_ts": None,   # market timestamp of the bad candle
    "last_reason": None,
}

# ---------------------------------------------------------------------------
# VIX all-rows-filtered tracking: records when _load_vix_candles found stored
# VIX rows but every one of them was dropped by the zero-volume filter, so the
# prediction router ran with no VIX data at all.  The time-based VIX staleness
# check cannot see this condition (the rows are recent, just unusable), so
# /api/healthz surfaces it separately.
# ---------------------------------------------------------------------------
_vix_all_filtered_stats: dict = {
    "count": 0,        # times all stored VIX rows were filtered, since startup
    "last_at": None,   # ISO timestamp of the most recent occurrence
    "rows_filtered": None,  # how many stored rows were dropped last time
}


def _record_vix_all_rows_filtered(rows_filtered: int) -> None:
    """Record that every stored VIX row was dropped by the zero-volume filter
    (never raises)."""
    try:
        _vix_all_filtered_stats["count"] += 1
        _vix_all_filtered_stats["last_at"] = datetime.utcnow().isoformat()
        _vix_all_filtered_stats["rows_filtered"] = int(rows_filtered)
        logger.warning(
            "vix_prediction_all_rows_filtered rows_filtered=%d count=%d — "
            "prediction is running without VIX data; macro signal degraded",
            rows_filtered, _vix_all_filtered_stats["count"],
        )
    except Exception as exc:
        logger.error("_record_vix_all_rows_filtered error: %s", exc)


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
_decision_filter = DecisionFilter()
_conviction = ConvictionEvaluator()
_long_model = LongTrendModel()
_short_model = ShortTrendModel()
_hold_engine = HoldTimePredictionEngine()


def _detect_zero_volume_bars(
    df_5min: pd.DataFrame,
) -> tuple["pd.Series[bool]", int, str]:
    """
    Identify 5-min bars whose volume is 0 (or missing).

    Zero-volume bars pass all OHLC consistency checks but can distort the
    liquidity score if they are included in the volume sum.  This helper
    returns the boolean mask so the caller can exclude those rows from
    liquidity-score computation while keeping them in the feature frame.

    Returns:
        (zero_vol_mask, zero_vol_count, dq_reason_fragment)
        ``dq_reason_fragment`` is ``""`` when no zero-volume bars are present.
    """
    if "volume" not in df_5min.columns or df_5min.empty:
        false_mask = pd.Series(False, index=df_5min.index, dtype=bool)
        return false_mask, 0, ""

    zero_vol_mask: "pd.Series[bool]" = df_5min["volume"].fillna(0).eq(0)
    zero_vol_count = int(zero_vol_mask.sum())
    if zero_vol_count == 0:
        return zero_vol_mask, 0, ""

    ts_col = df_5min.loc[zero_vol_mask, "timestamp"] if "timestamp" in df_5min.columns else pd.Series()
    ts_list = ts_col.tolist() if not ts_col.empty else []
    first_ts = ts_list[0] if ts_list else "unknown"
    first_ts_str = first_ts.isoformat() if hasattr(first_ts, "isoformat") else str(first_ts)
    reason = (
        f"zero_volume_bars: {zero_vol_count} 5-min bar(s) had volume=0 "
        f"and were excluded from liquidity score "
        f"(earliest ts={first_ts_str})"
    )
    return zero_vol_mask, zero_vol_count, reason


def _validate_ticker(ticker: str):
    """Validate ticker is VOO. Multi-ticker support will be added later."""
    if ticker.upper() != "VOO":
        raise HTTPException(
            status_code=400,
            detail=f"Ticker '{ticker}' not supported. Only 'VOO' is accepted in this version."
        )


def _parse_window(window: str) -> timedelta:
    """Parse window string like '3h', '24h', '7d', '30d', '3mo', '6mo' into timedelta.

    Lenient by design (see test_spec_gap_endpoints): unrecognized values fall
    back to 30 days rather than erroring, so older clients never break.
    """
    try:
        if window.endswith("mo"):
            return timedelta(days=int(window[:-2]) * 30)
        elif window.endswith("d"):
            return timedelta(days=int(window[:-1]))
        elif window.endswith("h"):
            return timedelta(hours=int(window[:-1]))
    except ValueError:
        pass
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

def _drop_invalid_ohlc(df: pd.DataFrame, timeframe: str = "daily") -> tuple[pd.DataFrame, bool, str]:
    """
    Remove internally-inconsistent OHLC rows from a loaded candle DataFrame.

    Called at prediction time so already-stored malformed candles (e.g. a
    yfinance ingest glitch) don't corrupt model features.

    Returns:
        (clean_df, data_quality_degraded, data_quality_reason)

    data_quality_degraded is True when any row was quarantined; the reason
    string describes the worst / most-recent bad candle.
    """
    from ingestion.ohlc_validator import filter_valid_ohlc

    try:
        if df.empty:
            return df, False, ""
        spike_threshold = (
            settings.DAILY_SPIKE_CLOSE_THRESHOLD
            if timeframe == "daily"
            else settings.SPIKE_CLOSE_THRESHOLD
        )
        valid_df, bad_df = filter_valid_ohlc(df, spike_threshold=spike_threshold)
        if bad_df.empty:
            return df, False, ""

        # Record each quarantined candle for healthz tracking
        for _, row in bad_df.iterrows():
            ts_val = row.get("timestamp", "unknown")
            ts_str = ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val)
            reason = str(row.get("ohlc_invalid_reason", "unknown"))
            _record_ohlc_quarantine(ts_str, reason)

        latest_bad = bad_df.iloc[-1]
        latest_ts = latest_bad.get("timestamp", "unknown")
        latest_ts_str = (
            latest_ts.isoformat() if hasattr(latest_ts, "isoformat") else str(latest_ts)
        )
        degraded_reason = (
            f"quarantined {len(bad_df)} malformed {timeframe} candle(s); "
            f"latest bad candle ts={latest_ts_str} "
            f"reason={latest_bad.get('ohlc_invalid_reason', 'unknown')}; "
            f"using last valid candle instead"
        )
        return valid_df, True, degraded_reason
    except Exception as exc:
        logger.error("_drop_invalid_ohlc error: %s", exc)
        return df, False, ""
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
    """Load VIX daily candles.

    ``^VIX`` is an index and Yahoo Finance reports its volume as zero. VIX
    validity is determined by its OHLC values, not by traded volume.
    """
    result = await session.execute(
        select(VixCandle)
        .order_by(desc(VixCandle.timestamp))
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()
    rows = list(reversed(rows))

    valid_rows = []
    for row in rows:
        issue = ohlc_validation_issue(row.open, row.high, row.low, row.close)
        if issue:
            logger.warning(
                "vix_prediction_invalid_ohlc_skipped ticker=%s timeframe=%s ts=%s issue=%s",
                row.ticker,
                row.timeframe,
                row.timestamp.isoformat()
                if hasattr(row.timestamp, "isoformat")
                else row.timestamp,
                issue,
            )
            continue
        valid_rows.append(row)

    return pd.DataFrame([{
        "timestamp": r.timestamp,
        "open": r.open, "high": r.high, "low": r.low, "close": r.close,
        "ticker": r.ticker
    } for r in valid_rows])


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

        # Zero-volume SPX bars are a yfinance glitch; they are removed at
        # startup by remove_invalid_spx_candles(), but a row may survive from
        # an older backup restore or a race before cleanup completes.  Filter
        # here so a bad close never corrupts the overnight-return signal.
        valid_rows = []
        for r in rows:
            vol = r.volume
            if vol is None or float(vol) == 0:
                logger.warning(
                    "spx_prediction_zero_volume_skipped ticker=%s timeframe=%s ts=%s",
                    r.ticker,
                    r.timeframe,
                    r.timestamp.isoformat() if hasattr(r.timestamp, "isoformat") else r.timestamp,
                )
                continue
            valid_rows.append(r)

        if not valid_rows:
            return pd.Series(dtype=float)

        return pd.Series(
            [r.close for r in valid_rows],
            index=pd.to_datetime([r.timestamp for r in valid_rows]),
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


async def _load_recent_confidence(
    session: AsyncSession, ticker: str, limit: int = 5
) -> List[dict]:
    """Load the most recent confidence snapshots for decision-filter checks."""
    try:
        result = await session.execute(
            select(ConfidenceHistory)
            .where(ConfidenceHistory.ticker == ticker)
            .order_by(desc(ConfidenceHistory.timestamp))
            .limit(limit)
        )
        rows = list(reversed(result.scalars().all()))
        return [
            {
                "long_buy_confidence": r.long_buy_confidence,
                "long_sell_confidence": r.long_sell_confidence,
                "short_buy_confidence": r.short_buy_confidence,
                "short_sell_confidence": r.short_sell_confidence,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("_load_recent_confidence error: %s", exc)
        return []


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
                        macro_override: bool, cycle_id: Optional[str] = None,
                        conviction_tier: Optional[str] = None,
                        conviction_reasons: Optional[list] = None):
    """Persist signal event to signal_history table."""
    import json as _json
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
        macro_override_applied=macro_override,
        conviction_tier=conviction_tier,
        conviction_reasons=(
            _json.dumps(conviction_reasons) if conviction_reasons else None
        ),
    )
    session.add(entry)
    await session.commit()
    return entry


async def _reliability_gate_allows(db: AsyncSession) -> tuple[bool, str]:
    """
    Reliability gate for push notifications: consult the reliability engine's
    recent win-rate metrics and suppress alerts when the system has been
    performing poorly.

    Rules:
      - Fewer than NOTIFY_RELIABILITY_MIN_CYCLES completed cycles in the
        window → allowed (a fresh system is never muted by lack of history).
      - win_rate < NOTIFY_MIN_WIN_RATE → suppressed.
    Errors always default to allowed (the gate must never block alerts due
    to its own failure).
    """
    try:
        from reliability_engine import compute_metrics, generate_trade_cycles

        min_cycles = int(getattr(settings, "NOTIFY_RELIABILITY_MIN_CYCLES", 5))
        min_win_rate = float(getattr(settings, "NOTIFY_MIN_WIN_RATE", 0.40))
        window = str(getattr(settings, "NOTIFY_RELIABILITY_WINDOW", "30d"))

        cycles = await generate_trade_cycles(db, window=window, persist=False)
        if len(cycles) < min_cycles:
            return True, (
                f"only {len(cycles)} cycle(s) in {window} "
                f"(< {min_cycles} required for gating)"
            )
        win_rate = float(compute_metrics(cycles).get("win_rate", 0.0))
        if win_rate < min_win_rate:
            return False, (
                f"win_rate={win_rate:.2%} over {len(cycles)} cycles in {window} "
                f"is below the {min_win_rate:.0%} reliability threshold"
            )
        return True, f"win_rate={win_rate:.2%} over {len(cycles)} cycles"
    except Exception as exc:
        logger.error("_reliability_gate_allows error (defaulting to allowed): %s", exc)
        return True, f"gate error (defaulting to allowed): {exc}"


async def _notify_all_devices_bg(
    signal_type: str,
    gauge_type: str,
    confidence: float,
    is_extended: bool,
    gap_type: str,
    liquidity_score: float,
    score: float,
    conviction_tier: Optional[str] = None,
) -> None:
    """
    Background task: send FCM push notifications to all registered device tokens.

    Each device's stored preferences are checked before firing:
      - Extended-hours signals are skipped when the device opted out.
      - Signals below the device's confidence threshold are silently skipped.
      - Devices in "high-conviction only" mode skip non-high-conviction signals.

    Uses its own DB session so it can run after the request session is closed.
    Errors are logged but never raise — this must not affect prediction responses.
    """
    from notifications.fcm import FCMNotifier

    try:
        session_factory = get_session_factory()
        async with session_factory() as db:
            # ── Reliability gate ────────────────────────────────────────────
            # Suppress all alerts when recent trade-cycle win rate is poor —
            # confidence alone is not enough when the system has been wrong.
            allowed, gate_reason = await _reliability_gate_allows(db)
            if not allowed:
                logger.warning(
                    "notification_suppressed_by_reliability_gate signal=%s "
                    "gauge=%s reason=%s", signal_type, gauge_type, gate_reason,
                )
                return

            result = await db.execute(select(DeviceToken))
            # Load all columns eagerly before the session closes.
            tokens = [
                {
                    "token": t.token,
                    "device_name": t.device_name,
                    "min_buy_threshold": t.min_buy_threshold,
                    "min_sell_threshold": t.min_sell_threshold,
                    "extended_hours_notifications": t.extended_hours_notifications,
                    "high_conviction_only": bool(
                        getattr(t, "high_conviction_only", False)
                    ),
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
            # Skip non-high-conviction signals for devices that opted into
            # high-conviction-only notifications.
            if (
                device["high_conviction_only"]
                and conviction_tier != TIER_HIGH_CONVICTION
            ):
                logger.debug(
                    "Skipping %s-tier notification for high-conviction-only device: %s",
                    conviction_tier or "untiered",
                    device["device_name"] or "unknown",
                )
                continue

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
                conviction_tier=conviction_tier,
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
                **dict(NEUTRAL_DEFAULTS),
                "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                "note": "No historical data yet. Run data ingestion first."
            }

        # OHLC integrity filter: drop any malformed candles stored in the DB
        # (e.g. a yfinance ingest glitch where high < open/close).
        daily_df, dq_degraded, dq_reason = _drop_invalid_ohlc(daily_df, timeframe="daily")
        if daily_df.empty:
            return {
                "score": 0, "signal": "neutral", "confidence": 0.5,
                "indicator_breakdown": {}, "ml_confidence": 0.5,
                "ml_fallback": True,
                "liquidity_score": 1.0, "gap_type": "none",
                "macro_override_applied": False,
                **dict(NEUTRAL_DEFAULTS),
                "data_quality_degraded": True,
                "data_quality_reason": dq_reason,
                "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                "note": "All recent daily candles failed OHLC integrity check."
            }

        # Zero-volume daily bar detection: a yfinance glitch may return a daily
        # bar with volume=0 that passes all OHLC consistency checks but would
        # corrupt any volume-based long-trend features (e.g. OBV, volume EMA).
        # Exclude those bars from the frame so they cannot distort predictions.
        zero_vol_mask, zero_vol_count, zv_reason = _detect_zero_volume_bars(daily_df)
        if zero_vol_count > 0:
            daily_df = daily_df[~zero_vol_mask].reset_index(drop=True)
            if dq_degraded:
                dq_reason = dq_reason + "; " + zv_reason
            else:
                dq_reason = zv_reason
                dq_degraded = True
            logger.warning(
                "zero_volume_bars count=%d timeframe=daily",
                zero_vol_count,
            )
            if daily_df.empty:
                return {
                    "score": 0, "signal": "neutral", "confidence": 0.5,
                    "indicator_breakdown": {}, "ml_confidence": 0.5,
                    "ml_fallback": True,
                    "liquidity_score": 1.0, "gap_type": "none",
                    "macro_override_applied": False,
                    **dict(NEUTRAL_DEFAULTS),
                    "data_quality_degraded": True,
                    "data_quality_reason": dq_reason,
                    "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                    "note": "All recent daily candles had zero volume."
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

        # Update in-memory confidence cache so the divergence check has the
        # latest long-trend buy confidence available.
        global _last_long_buy_conf, _last_long_sell_conf
        _last_long_buy_conf = float(long_buy_conf)
        _last_long_sell_conf = float(long_sell_conf)

        # Apply VOO-only decision-layer filters after the gauge.
        confidence_history = await _load_recent_confidence(session, ticker, limit=5)
        confidence_history.append({
            "long_buy_confidence": _last_long_buy_conf,
            "short_buy_confidence": _last_short_buy_conf,
        })
        decision = _decision_filter.evaluate(
            signal_type=result["signal"],
            score=result["score"],
            ml_confidence=ml_confidence,
            indicators=indicators,
            latest_candle=latest.to_dict(),
            liquidity_score=1.0,
            gap_momentum=None,
            confidence_history=confidence_history,
            data_quality_degraded=dq_degraded,
        )
        final_signal = decision["final_signal"]
        is_candidate = decision.get("is_candidate", False)
        candidate_signal = decision.get("candidate_signal")
        notify_confidence = min(
            1.0,
            max(
                0.0,
                abs(result["score"]) / 100.0
                + decision.get("priority_boost", 0.0)
                - decision.get("decision_penalty", 0.0),
            ),
        )

        # Conviction tier (label only — never suppresses the signal).
        # Candidates are evaluated against the raw direction but capped at
        # opportunity; they are never stored or notified, so the tier is
        # purely informational display metadata.
        conviction_signal = candidate_signal if is_candidate else final_signal
        conviction = _conviction.evaluate(
            signal_type=conviction_signal,
            gauge_type="long",
            volatility_regime=decision.get("volatility_regime", "calm"),
            cycle_quality_score=decision.get("cycle_quality_score", 0.5),
            ml_confidence=ml_confidence,
            ml_fallback=ml_fallback,
            long_score=result["score"],
            short_score=_last_short_score,
            tier_cap="opportunity" if is_candidate else decision.get("conviction_tier_cap"),
            tier_cap_reason=(
                "Signal is a candidate — directional hint only, not executable."
                if is_candidate else decision.get("reason")
            ),
        )

        # Persist confidence history
        await _store_confidence(session, ticker,
                                long_buy=long_buy_conf, long_sell=long_sell_conf,
                                # Prediction endpoints run independently. Keep
                                # the latest value from the other gauge instead
                                # of writing a misleading zero into every row.
                                short_buy=_last_short_buy_conf,
                                short_sell=_last_short_sell_conf,
                                session_type=session_type, is_extended=is_extended)

        # Persist signal if actionable, then push notification in background.
        # Candidates are NOT stored (to avoid false BUY→SELL cycles) and do
        # NOT trigger push notifications.
        if final_signal in ("buy", "sell"):
            await _store_signal(
                session, ticker,
                signal_type=final_signal, gauge_type="long",
                confidence=abs(result["score"]) / 100.0,
                session_type=session_type, is_extended=is_extended,
                gap_type=str(latest.get("gap_type", "none")),
                liquidity_score=1.0, macro_override=False,
                conviction_tier=conviction["tier"],
                conviction_reasons=conviction["reasons"],
            )
            asyncio.create_task(_notify_all_devices_bg(
                signal_type=final_signal,
                gauge_type="long",
                confidence=notify_confidence,
                is_extended=is_extended,
                gap_type=str(latest.get("gap_type", "none")),
                liquidity_score=1.0,
                score=result["score"],
                conviction_tier=conviction["tier"],
            ))

        return {
            "score": result["score"],
            "signal": final_signal,
            "is_candidate": is_candidate,
            "candidate_signal": candidate_signal,
            "confidence": result["confidence"],
            **normalize_gauge_output(result["score"]),
            "indicator_breakdown": result.get("breakdown", {}),
            "ml_confidence": ml_confidence,
            "ml_fallback": ml_fallback,
            "liquidity_score": 1.0,
            "gap_type": str(latest.get("gap_type", "none")),
            "macro_override_applied": False,
            "decision_filter_applied": True,
            "decision_filter_reason": decision.get("reason", ""),
            "cycle_quality_score": decision.get("cycle_quality_score", 0.5),
            "volatility_regime": decision.get("volatility_regime", "calm"),
            "liquidity_class": decision.get("liquidity_class", "normal"),
            "confidence_momentum": decision.get("confidence_momentum", 0.0),
            "conviction_tier": conviction["tier"] if not is_candidate else None,
            "conviction_reasons": conviction["reasons"] if not is_candidate else [],
            "candidate_conviction_tier": conviction["tier"] if is_candidate else None,
            "candidate_conviction_reasons": conviction["reasons"] if is_candidate else [],
            "data_quality_degraded": dq_degraded,
            "data_quality_reason": dq_reason,
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
                **dict(NEUTRAL_DEFAULTS),
                "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                "note": "No 5-min data yet. Run data ingestion first."
            }

        # OHLC integrity filter: drop any malformed candles stored in the DB.
        df_5min, dq_degraded, dq_reason = _drop_invalid_ohlc(df_5min, timeframe="5min")
        if df_5min.empty:
            return {
                "score": 0, "signal": "neutral", "confidence": 0.5,
                "indicator_breakdown": {}, "ml_confidence": 0.5,
                "ml_fallback": True,
                "liquidity_score": 1.0, "gap_type": "none",
                "macro_override_applied": False,
                **dict(NEUTRAL_DEFAULTS),
                "data_quality_degraded": True,
                "data_quality_reason": dq_reason,
                "timestamp": datetime.utcnow().isoformat(), "ticker": ticker,
                "note": "All recent 5-min candles failed OHLC integrity check."
            }

        # Zero-volume bar detection: these bars pass OHLC checks but have
        # volume=0 (yfinance glitch or thin extended-hours window).  Exclude
        # them from the liquidity score so a single glitch bar cannot
        # artificially suppress valid signals via the low-liquidity weight
        # reduction.  The bars are kept in df_5min for feature computation
        # (their price data may still be reliable).
        zero_vol_mask, zero_vol_count, zv_reason = _detect_zero_volume_bars(df_5min)
        if zero_vol_count > 0:
            if dq_degraded:
                dq_reason = dq_reason + "; " + zv_reason
            else:
                dq_reason = zv_reason
                dq_degraded = True
            logger.warning(
                "zero_volume_bars count=%d timeframe=5min",
                zero_vol_count,
            )

        # Choose the "latest" candle for session context (session_type, gap_type,
        # is_extended).  When the most recent bar has volume=0 (a glitch bar) we
        # fall back to the last non-zero-volume bar so that a single empty bar
        # cannot corrupt the signal context.  If all bars are zero-volume we keep
        # the raw last row as a best-effort fallback (extremely rare edge case).
        if zero_vol_count > 0 and bool(zero_vol_mask.iloc[-1]):
            valid_rows = df_5min[~zero_vol_mask]
            latest = valid_rows.iloc[-1] if not valid_rows.empty else df_5min.iloc[-1]
        else:
            latest = df_5min.iloc[-1]
        is_extended = bool(latest.get("is_extended_hours", False))
        session_type = str(latest.get("session_type", "regular"))
        gap_type = str(latest.get("gap_type", "none"))
        # Gap follow-through momentum (additive; None when no gap / no data yet)
        gap_momentum = _compute_gap_momentum_from_df(df_5min)

        # Compute liquidity score using only non-zero-volume bars so that a
        # glitch bar with volume=0 cannot drag the score toward zero.
        df_5min_for_liq = df_5min[~zero_vol_mask] if zero_vol_count > 0 else df_5min
        regular_mask = df_5min_for_liq["session_type"] == "regular"
        extended_mask = df_5min_for_liq["is_extended_hours"] == True
        regular_vol = float(df_5min_for_liq.loc[regular_mask, "volume"].sum()) if regular_mask.any() else 1.0
        extended_vol = float(df_5min_for_liq.loc[extended_mask, "volume"].sum()) if extended_mask.any() else 0.0
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
        # A healthy calibrated short model uses its rare-event base rate as
        # the ML-neutral point.  Missing/stale/error fallbacks keep 0.5 so a
        # fallback cannot accidentally add a bullish or bearish bias.
        ml_neutral_probability = (
            _short_model.get_neutral_probability() if not ml_fallback else 0.5
        )
        result = _short_gauge.compute_score(
            indicators, ml_confidence,
            is_extended=is_extended,
            liquidity_score=liquidity_score,
            gap_type=gap_type,
            age_in_minutes=0,
            gap_momentum=gap_momentum,
            neutral_probability=ml_neutral_probability,
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

        # Update in-memory confidence cache so the divergence check has the
        # latest short-trend buy confidence available.
        global _last_short_buy_conf, _last_short_sell_conf
        _last_short_buy_conf = float(short_buy_conf)
        _last_short_sell_conf = float(short_sell_conf)

        # Apply VOO-only decision-layer filters after the macro override.
        confidence_history = await _load_recent_confidence(session, ticker, limit=5)
        confidence_history.append({
            "long_buy_confidence": _last_long_buy_conf,
            "short_buy_confidence": _last_short_buy_conf,
        })
        decision = _decision_filter.evaluate(
            signal_type=final_signal,
            score=result["score"],
            ml_confidence=ml_confidence,
            indicators=indicators,
            latest_candle=latest.to_dict(),
            liquidity_score=liquidity_score,
            gap_momentum=gap_momentum,
            confidence_history=confidence_history,
            data_quality_degraded=dq_degraded,
        )
        final_signal = decision["final_signal"]
        is_candidate = decision.get("is_candidate", False)
        candidate_signal = decision.get("candidate_signal")
        notify_confidence = min(
            1.0,
            max(
                0.0,
                abs(result["score"]) / 100.0
                + decision.get("priority_boost", 0.0)
                - decision.get("decision_penalty", 0.0),
            ),
        )

        # Conviction tier (label only — never suppresses the signal).
        # Candidates are evaluated against the raw direction but capped at
        # opportunity; they are never stored or notified, so the tier is
        # purely informational display metadata.
        conviction_signal = candidate_signal if is_candidate else final_signal
        conviction = _conviction.evaluate(
            signal_type=conviction_signal,
            gauge_type="short",
            volatility_regime=decision.get("volatility_regime", "calm"),
            cycle_quality_score=decision.get("cycle_quality_score", 0.5),
            ml_confidence=ml_confidence,
            ml_fallback=ml_fallback,
            long_score=_last_long_score,
            short_score=result["score"],
            tier_cap="opportunity" if is_candidate else decision.get("conviction_tier_cap"),
            tier_cap_reason=(
                "Signal is a candidate — directional hint only, not executable."
                if is_candidate else decision.get("reason")
            ),
        )

        # Persist confidence history
        await _store_confidence(session, ticker,
                                # Preserve the latest long-gauge values; the
                                # long and short endpoints are called
                                # separately but history rows represent a
                                # combined chart snapshot.
                                long_buy=_last_long_buy_conf,
                                long_sell=_last_long_sell_conf,
                                short_buy=short_buy_conf, short_sell=short_sell_conf,
                                session_type=session_type, is_extended=is_extended)

        # Persist signal if actionable, then push notification in background.
        # Candidates are NOT stored (to avoid false BUY→SELL cycles) and do
        # NOT trigger push notifications.
        if final_signal in ("buy", "sell"):
            await _store_signal(
                session, ticker,
                signal_type=final_signal, gauge_type="short",
                confidence=abs(result["score"]) / 100.0,
                session_type=session_type, is_extended=is_extended,
                gap_type=gap_type, liquidity_score=liquidity_score,
                macro_override=macro_override_applied,
                conviction_tier=conviction["tier"],
                conviction_reasons=conviction["reasons"],
            )
            asyncio.create_task(_notify_all_devices_bg(
                signal_type=final_signal,
                gauge_type="short",
                confidence=notify_confidence,
                is_extended=is_extended,
                gap_type=gap_type,
                liquidity_score=liquidity_score,
                score=result["score"],
                conviction_tier=conviction["tier"],
            ))

        return {
            "score": result["score"],
            "signal": final_signal,
            "is_candidate": is_candidate,
            "candidate_signal": candidate_signal,
            "confidence": result["confidence"],
            # Downgrade display_signal to HOLD when the macro override forced
            # the filtered signal neutral — the bias label must never
            # contradict an override-suppressed signal.
            **reconcile_display_signal(
                normalize_gauge_output(result["score"]),
                final_signal, macro_override_applied,
            ),
            "indicator_breakdown": result.get("breakdown", {}),
            "ml_confidence": ml_confidence,
            "ml_neutral_probability": ml_neutral_probability,
            "ml_fallback": ml_fallback,
            "liquidity_score": liquidity_score,
            "gap_type": gap_type,
            "gap_momentum": gap_momentum,
            "macro_override_applied": macro_override_applied,
            "macro_override_reason": override_result.get("reason", ""),
            "decision_filter_applied": True,
            "decision_filter_reason": decision.get("reason", ""),
            "cycle_quality_score": decision.get("cycle_quality_score", 0.5),
            "volatility_regime": decision.get("volatility_regime", "calm"),
            "liquidity_class": decision.get("liquidity_class", "normal"),
            "confidence_momentum": decision.get("confidence_momentum", 0.0),
            "conviction_tier": conviction["tier"] if not is_candidate else None,
            "conviction_reasons": conviction["reasons"] if not is_candidate else [],
            "candidate_conviction_tier": conviction["tier"] if is_candidate else None,
            "candidate_conviction_reasons": conviction["reasons"] if is_candidate else [],
            "data_quality_degraded": dq_degraded,
            "data_quality_reason": dq_reason,
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
def _downsample_uniform(rows: list, max_points: int) -> list:
    """Uniformly downsample rows to at most max_points, keeping both endpoints.

    Rows are assumed ordered (newest-first here); sampling preserves order and
    always includes the first and last row so the full window span is retained.
    """
    n = len(rows)
    if n <= max_points:
        return rows
    step = (n - 1) / (max_points - 1)
    return [rows[round(i * step)] for i in range(max_points)]


# GET /confidence_history
# ---------------------------------------------------------------------------
@router.get("/confidence_history")
async def confidence_history(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="7d"),
    session: AsyncSession = Depends(get_session)
):
    """Return confidence history for the given ticker and time window.

    Long windows (3mo/6mo) can hold far more than the response cap of 1000
    points. Instead of truncating to the newest 1000 rows (which would silently
    drop the older part of the window), rows are downsampled uniformly across
    the full window so the chart always spans the requested period.
    """
    _validate_ticker(ticker)
    since = datetime.utcnow() - _parse_window(window)
    result = await session.execute(
        select(ConfidenceHistory)
        .where(and_(
            ConfidenceHistory.ticker == ticker,
            ConfidenceHistory.timestamp >= since
        ))
        .order_by(desc(ConfidenceHistory.timestamp))
    )
    rows = result.scalars().all()
    rows = _downsample_uniform(rows, max_points=1000)
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
            "macro_override_applied": r.macro_override_applied,
            "conviction_tier": r.conviction_tier,
            "conviction_reasons": _parse_reasons(r.conviction_reasons),
        }
        for r in rows
    ]


def _parse_reasons(raw: Optional[str]) -> list:
    """Decode a JSON-encoded conviction_reasons column (never raises)."""
    if not raw:
        return []
    try:
        import json as _json
        parsed = _json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


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
                "is_extended_hours": sig.is_extended_hours,
                "conviction_tier": sig.conviction_tier,
                "conviction_reasons": _parse_reasons(sig.conviction_reasons),
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
                "is_extended_hours": sig.is_extended_hours,
                "conviction_tier": sig.conviction_tier,
                "conviction_reasons": _parse_reasons(sig.conviction_reasons),
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
# GET /model_performance  (Model Performance Dashboard)
# ---------------------------------------------------------------------------
@router.get("/model_performance")
async def model_performance(
    ticker: str = Query(default="VOO"),
    period: str = Query(default="day"),
    window: str = Query(default="90d"),
    confidence_min: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    confidence_max: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_session)
):
    """
    Model performance report: period-bucketed BUY precision, confidence
    buckets, calibration curve, cumulative P&L, streaks, missed rallies,
    session/VIX breakdowns, and retrain accuracy history.

    Safe empty shapes are returned when no trade data exists (never a 500).
    """
    _validate_ticker(ticker)
    if period not in ("day", "week", "month"):
        raise HTTPException(
            status_code=400,
            detail=f"period must be one of day, week, month (got '{period}')",
        )
    from performance_engine import get_model_performance
    return await get_model_performance(
        session,
        ticker=ticker,
        period=period,
        window=window,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
    )


@router.get("/tier_track_record")
async def tier_track_record(
    ticker: str = Query(default="VOO"),
    window: str = Query(default="90d"),
    session: AsyncSession = Depends(get_session)
):
    """
    Realized performance per conviction tier (win rate, average return per
    completed BUY→SELL cycle), over a selectable window: '30d', '90d', 'all'.

    Tiers with fewer than min_sample_size completed cycles report null
    win_rate/avg_return_percent with sufficient_sample=false so clients can
    show "not enough signals yet" instead of a misleading percentage.
    Cycles missing price data are excluded (excluded_price_data_absent).
    """
    _validate_ticker(ticker)
    from performance_engine import get_tier_track_record, TIER_WINDOWS
    if window not in TIER_WINDOWS:
        raise HTTPException(
            status_code=400,
            detail=f"window must be one of {', '.join(TIER_WINDOWS)} (got '{window}')",
        )
    return await get_tier_track_record(session, ticker=ticker, window=window)


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
        active_accuracy = None
        try:
            result = await session.execute(
                select(ModelMetadata)
                .where(ModelMetadata.model_name == name)
                .order_by(ModelMetadata.trained_at.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            if row is not None:
                last_trained_at = row.trained_at.isoformat() if row.trained_at else None
                active_accuracy = row.accuracy
        except Exception as exc:
            logger.error("healthz: %s metadata lookup failed: %s", name, exc)

        status = training_status.get(name, {})
        failed = status.get("success") is False
        rolled_back = bool(status.get("rolled_back"))
        if status.get("success") is True:
            last_retrain_outcome = "success"
        elif failed:
            last_retrain_outcome = "rolled_back" if rolled_back else "failed"
        else:
            last_retrain_outcome = None
        consecutive_failures = status.get("consecutive_failures") or 0
        training_stuck = consecutive_failures >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD
        fallback_stats = _ml_fallback_stats.get(name, {})
        persisted = persisted_fallbacks.get(name, {})
        if failed or neutral or fallback_stats.get("count", 0) > 0:
            degraded = True

        models[name] = {
            "last_training_success": status.get("success"),
            "last_retrain_outcome": last_retrain_outcome,
            "last_retrain_rolled_back": rolled_back,
            "last_retrain_attempted_accuracy": status.get("accuracy"),
            "active_model_accuracy": (
                active_accuracy
                if active_accuracy is not None
                else status.get("last_success_accuracy")
            ),
            "consecutive_training_failures": consecutive_failures,
            "training_stuck": training_stuck,
            "last_training_error": status.get("error"),
            "last_training_attempted_at": status.get("attempted_at"),
            "last_training_accuracy": status.get("accuracy"),
            # Which metric the headline accuracy actually is. 'train' means
            # the purged walk-forward OOS evaluation could not run (too few
            # rows), so the reported accuracy is NOT an honest OOS number.
            "last_training_accuracy_metric": status.get("accuracy_metric"),
            "active_model_accuracy_metric": status.get(
                "last_success_accuracy_metric"
            ),
            "walk_forward_evaluation_skipped": (
                status.get("accuracy_metric") == "train"
            ),
            "last_trained_at": last_trained_at,
            "neutral_fallback": neutral,
            "ml_fallback_count": fallback_stats.get("count", 0),
            "ml_fallback_last_at": fallback_stats.get("last_at"),
            "ml_fallback_last_reason": fallback_stats.get("last_reason"),
            "ml_fallback_total_count": persisted.get("total_count", 0),
            "ml_fallback_total_last_at": persisted.get("last_at"),
            "ml_fallback_total_last_reason": persisted.get("last_reason"),
        }

        # Walk-forward calibration report (long-trend only): honest OOS
        # accuracy, Brier score, and reliability bins from the last train.
        if name == "long_trend":
            try:
                from ml.calibration import get_calibration_report
                models[name]["calibration"] = get_calibration_report()
            except Exception as exc:
                logger.error("healthz: calibration report lookup failed: %s", exc)
                models[name]["calibration"] = None

        # Purged walk-forward evaluation report (short-trend): honest OOS
        # accuracy replaces the leakage-inflated train accuracy.
        if name == "short_trend":
            try:
                from ml.calibration import get_walkforward_report
                models[name]["walk_forward"] = get_walkforward_report("short_trend")
            except Exception as exc:
                logger.error("healthz: walk-forward report lookup failed: %s", exc)
                models[name]["walk_forward"] = None

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

    # ── Push notification readiness ────────────────────────────────────────
    # Keep this operational summary secret-free: it reports configuration
    # state and counts, never the FCM credential or device-token contents.
    notification_readiness = {
        "fcm_server_configured": bool(settings.FCM_SERVER_KEY),
        "registered_devices": 0,
        "ready": False,
        "blockers": [],
    }
    try:
        device_count = await session.scalar(
            select(func.count(DeviceToken.id))
        ) or 0
        notification_readiness["registered_devices"] = int(device_count)
        if not settings.FCM_SERVER_KEY:
            notification_readiness["blockers"].append(
                "FCM_SERVER_KEY is not configured"
            )
        if device_count == 0:
            notification_readiness["blockers"].append(
                "No Android device token is registered"
            )
        notification_readiness["ready"] = (
            bool(settings.FCM_SERVER_KEY) and device_count > 0
        )
    except Exception as exc:
        logger.error("healthz: notification readiness lookup failed: %s", exc)
        notification_readiness["blockers"].append(
            "Could not inspect notification registration state"
        )

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
            if info.get("last_retrain_rolled_back"):
                attempted = info.get("last_retrain_attempted_accuracy")
                details = []
                if info["last_training_error"]:
                    details.append(str(info["last_training_error"]))
                if attempted is not None:
                    details.append(f"attempted accuracy {attempted:.4f}")
                alerts.append(
                    f"{name}: last retrain was rolled back — model restored to "
                    "last known-good version"
                    + (f" ({'; '.join(details)})" if details else "")
                )
            else:
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
        if info.get("walk_forward_evaluation_skipped"):
            acc = info.get("last_training_accuracy")
            alerts.append(
                f"{name}: walk-forward evaluation could not run on the last "
                "retrain (too few rows) — reported accuracy"
                + (f" {acc:.4f}" if isinstance(acc, (int, float)) else "")
                + " is TRAIN accuracy, not an honest out-of-sample metric"
            )
        if info["ml_fallback_count"] > 0:
            alerts.append(
                f"{name}: served neutral-fallback predictions {info['ml_fallback_count']} time(s) "
                f"since startup (last: {info['ml_fallback_last_reason']} at {info['ml_fallback_last_at']})"
            )

    # ── VIX all-rows-filtered summary (zero-volume wipe) ─────────────────
    vix_zero_volume_filter = dict(_vix_all_filtered_stats)
    if _vix_all_filtered_stats["count"] > 0:
        degraded = True
        alerts.append(
            "vix_prediction_all_rows_filtered: predictions ran without VIX data "
            f"{_vix_all_filtered_stats['count']} time(s) since startup — every stored "
            f"VIX row was dropped by the zero-volume filter "
            f"(last: {_vix_all_filtered_stats['rows_filtered']} row(s) filtered "
            f"at {_vix_all_filtered_stats['last_at']})"
        )

    # ── OHLC data-quality quarantine summary ─────────────────────────────
    ohlc_quarantine = dict(_ohlc_quarantine_stats)
    if _ohlc_quarantine_stats["count"] > 0:
        degraded = True
        alerts.append(
            f"ohlc_invalid: {_ohlc_quarantine_stats['count']} malformed candle(s) quarantined "
            f"since startup (last ts={_ohlc_quarantine_stats['last_ts']} "
            f"reason={_ohlc_quarantine_stats['last_reason']} "
            f"at {_ohlc_quarantine_stats['last_at']})"
        )

    from database.cleanup_state import is_cleanup_pending
    from database.startup_state import (
        get_startup_status, get_startup_error, get_retrain_skipped_at_startup,
    )
    cleanup_pending = is_cleanup_pending()
    startup_status = get_startup_status()
    startup_error = get_startup_error()
    retrain_skipped_at_startup = get_retrain_skipped_at_startup()

    if startup_status == "degraded":
        degraded = True
        alerts.append(
            "startup: pipeline initialization failed — jobs unblocked in a degraded state"
            + (f" ({startup_error})" if startup_error else "")
        )

    if retrain_skipped_at_startup:
        degraded = True
        alerts.append(
            "startup: retrain was skipped because pipeline initialization did not complete "
            "cleanly — models may be stale; retrain will run on the next weekly schedule"
        )

    return {
        "status": "degraded" if degraded else "ok",
        "ticker": "VOO",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "NovaCycle API",
        "startup_status": startup_status,
        "startup_error": startup_error,
        "retrain_skipped_at_startup": retrain_skipped_at_startup,
        "cleanup_pending": cleanup_pending,
        "models": models,
        "spx_futures": spx_data,
        "vix": vix_data,
        "voo_5min": fivemin_data,
        "voo_5min_recovery": fivemin_recovery,
        "notifications": notification_readiness,
        "ohlc_quarantine": ohlc_quarantine,
        "vix_zero_volume_filter": vix_zero_volume_filter,
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

# ---------------------------------------------------------------------------
# POST /admin/cleanup_malformed_candles  (operator-only)
# ---------------------------------------------------------------------------
@router.post("/admin/cleanup_malformed_candles", dependencies=[Depends(_require_admin_token)])
async def cleanup_malformed_candles_endpoint(session: AsyncSession = Depends(get_session)):
    """
    Operator action: scan voo_candles, vix_candles, and spx_candles for rows
    that violate OHLC consistency rules (high < open, low > close, etc.) and
    permanently delete them.

    This is the on-demand counterpart to the automatic startup cleanup.  Run it
    after a data-quality incident to clear debris without restarting the server.

    On success the in-memory ohlc_quarantine counter is reset to zero so
    /api/healthz immediately reflects a clean state.  The counter will only
    increment again when a *new* malformed candle is encountered at prediction
    time (i.e. a live ingest glitch, not historical debris).

    Returns a structured summary:
        rows_found, rows_removed, tables_affected, timeframes_affected, details
    """
    from database.ohlc_cleanup import remove_malformed_candles

    try:
        summary = await remove_malformed_candles(session)
        await session.commit()
    except Exception as exc:
        logger.error("cleanup_malformed_candles failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {exc}")

    # Reset the in-memory quarantine counter so healthz shows count=0 immediately.
    # The counter will re-increment only if new live bad candles appear.
    _ohlc_quarantine_stats["count"] = 0
    _ohlc_quarantine_stats["last_at"] = None
    _ohlc_quarantine_stats["last_ts"] = None
    _ohlc_quarantine_stats["last_reason"] = None

    logger.info(
        "cleanup_malformed_candles triggered by operator: rows_found=%d rows_removed=%d "
        "tables=%s timeframes=%s",
        summary["rows_found"], summary["rows_removed"],
        summary["tables_affected"], summary["timeframes_affected"],
    )
    return {
        "status": "ok",
        "cleanup_at": datetime.now(timezone.utc).isoformat(),
        **summary,
    }


def _record_ohlc_quarantine(ts: str, reason: str) -> None:
    """Record that a malformed OHLC candle was detected at prediction time.

    Updates the in-memory tracker so /healthz can surface the condition.
    Never raises.
    """
    try:
        _ohlc_quarantine_stats["count"] += 1
        _ohlc_quarantine_stats["last_at"] = datetime.utcnow().isoformat()
        _ohlc_quarantine_stats["last_ts"] = ts
        _ohlc_quarantine_stats["last_reason"] = str(reason)[:300]
        logger.warning(
            "prediction_ohlc_invalid ts=%s reason=%s total_quarantined=%d",
            ts, reason, _ohlc_quarantine_stats["count"],
        )
    except Exception as exc:
        logger.error("_record_ohlc_quarantine error: %s", exc)
