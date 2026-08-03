"""
Model Performance Engine
========================
Computes the /api/model_performance payload:

  - period-bucketed summaries (day / ISO week / month)
  - confidence-band filtering + Low/Medium/High buckets
  - 10-point calibration curve
  - cumulative P&L series (compounding)
  - win/loss streaks
  - missed-rally detection (HOLD gaps where price rose past the short-trend
    target of +0.3% within 12 five-minute bars)
  - session-type and VIX-regime breakdowns
  - retrain accuracy history from ModelMetadata

All functions return safe empty shapes when no data exists — never a 500.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import ModelMetadata, SignalHistory, VooCandle
from reliability_engine import (
    _build_return_distribution,
    _classify_vix_regime,
    _load_vix_close_series,
    _lookup_vix_close,
    _parse_window,
    generate_trade_cycles,
)

logger = logging.getLogger(__name__)

# Shared rally-event definition (rally_event.py): >0.3% rise within 12
# five-minute bars.  The short model trains on the SAME event, so a missed
# rally here is exactly an event the model was supposed to predict.
from rally_event import RALLY_HORIZON_BARS, RALLY_RISE_PERCENT

MISSED_RALLY_RISE_PERCENT = RALLY_RISE_PERCENT
MISSED_RALLY_BARS = RALLY_HORIZON_BARS

# Confidence bands (fractions 0..1)
CONFIDENCE_BANDS = {
    "low": (0.0, 0.4),
    "medium": (0.4, 0.7),
    "high": (0.7, 1.0),
}

VALID_PERIODS = ("day", "week", "month")

# Conviction tiers tracked by the tier track record.  "untiered" collects
# cycles whose BUY signal predates tiering (conviction_tier is NULL).
TIER_KEYS = ("high_conviction", "opportunity", "untiered")

# Minimum number of completed cycles a tier needs before we surface a win
# rate.  Below this, percentages from tiny samples mislead more than inform.
MIN_TIER_SAMPLE = 5

# Selectable windows for the tier track record endpoint.
TIER_WINDOWS = ("30d", "90d", "all")


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-testable without a DB)
# ─────────────────────────────────────────────────────────────────────────────

def filter_cycles_by_confidence(
    cycles: List[Dict[str, Any]],
    confidence_min: Optional[float],
    confidence_max: Optional[float],
) -> List[Dict[str, Any]]:
    """Keep cycles whose confidence_at_buy falls in the band.

    Band semantics are half-open [min, max) so adjacent bands never overlap
    (0.4 belongs to medium, 0.7 to high), except the top of the scale: when
    max >= 1.0 the band is closed so a confidence of exactly 1.0 is included.
    This matches compute_confidence_buckets and the Android local filter.
    """
    if confidence_min is None and confidence_max is None:
        return cycles
    lo = confidence_min if confidence_min is not None else 0.0
    hi = confidence_max if confidence_max is not None else 1.0
    out = []
    for c in cycles:
        conf = c.get("confidence_at_buy")
        if conf is None:
            continue
        conf = float(conf)
        if lo <= conf < hi or (hi >= 1.0 and conf == 1.0):
            out.append(c)
    return out


def compute_confidence_buckets(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group cycles into Low/Medium/High confidence bands."""
    buckets: Dict[str, Any] = {}
    for name, (lo, hi) in CONFIDENCE_BANDS.items():
        group = [
            c for c in cycles
            if c.get("confidence_at_buy") is not None
            and (lo <= float(c["confidence_at_buy"]) < hi
                 or (hi == 1.0 and float(c["confidence_at_buy"]) == 1.0))
        ]
        wins = sum(1 for c in group if (c.get("return_percent") or 0.0) > 0.0)
        n = len(group)
        buckets[name] = {
            "trade_count": n,
            "win_rate": (wins / n) if n else 0.0,
            "avg_return_percent": (
                sum(float(c.get("return_percent") or 0.0) for c in group) / n
            ) if n else 0.0,
        }
    return buckets


def compute_calibration_curve(cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """10-point calibration curve: stated confidence bucket vs actual win rate."""
    points: List[Dict[str, Any]] = []
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        group = [
            c for c in cycles
            if c.get("confidence_at_buy") is not None
            and (lo <= float(c["confidence_at_buy"]) < hi
                 or (i == 9 and float(c["confidence_at_buy"]) == 1.0))
        ]
        n = len(group)
        wins = sum(1 for c in group if (c.get("return_percent") or 0.0) > 0.0)
        points.append({
            "confidence_mid": round(lo + 0.05, 2),
            "actual_win_rate": (wins / n) if n else None,
            "trade_count": n,
        })
    return points


def compute_cumulative_pnl(cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chronological compounding cumulative return series."""
    ordered = sorted(
        [c for c in cycles if c.get("sell_timestamp") is not None],
        key=lambda c: c["sell_timestamp"],
    )
    series: List[Dict[str, Any]] = []
    factor = 1.0
    for c in ordered:
        r = float(c.get("return_percent") or 0.0)
        factor *= (1.0 + r / 100.0)
        ts = c["sell_timestamp"]
        series.append({
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            "cumulative_return_percent": (factor - 1.0) * 100.0,
        })
    return series


def compute_streaks(cycles: List[Dict[str, Any]]) -> Dict[str, int]:
    """Current and longest win/loss streaks (chronological order)."""
    ordered = sorted(
        [c for c in cycles if c.get("sell_timestamp") is not None],
        key=lambda c: c["sell_timestamp"],
    )
    longest_win = longest_loss = 0
    cur_win = cur_loss = 0
    for c in ordered:
        if (c.get("return_percent") or 0.0) > 0.0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        longest_win = max(longest_win, cur_win)
        longest_loss = max(longest_loss, cur_loss)
    return {
        "current_win": cur_win,
        "current_loss": cur_loss,
        "longest_win": longest_win,
        "longest_loss": longest_loss,
    }


def _period_label(ts: datetime, period: str) -> Tuple[str, datetime]:
    """Return (label, bucket_start) for a timestamp under a period granularity."""
    if period == "week":
        iso = ts.isocalendar()
        label = f"{iso[0]}-W{iso[1]:02d}"
        start = datetime.fromisocalendar(iso[0], iso[1], 1)
    elif period == "month":
        label = ts.strftime("%Y-%m")
        start = datetime(ts.year, ts.month, 1)
    else:  # day
        label = ts.strftime("%Y-%m-%d")
        start = datetime(ts.year, ts.month, ts.day)
    return label, start


def bucket_cycles_by_period(
    cycles: List[Dict[str, Any]],
    period: str,
    missed_rally_timestamps: Optional[List[datetime]] = None,
    accuracy_history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Group cycles by calendar day / ISO week / calendar month."""
    if period not in VALID_PERIODS:
        period = "day"
    buckets: Dict[str, Dict[str, Any]] = {}
    for c in cycles:
        ts = c.get("buy_timestamp")
        if not isinstance(ts, datetime):
            continue
        label, start = _period_label(ts, period)
        b = buckets.setdefault(label, {
            "label": label, "start": start, "cycles": [],
        })
        b["cycles"].append(c)

    # Missed rallies per bucket
    rally_counts: Dict[str, int] = {}
    for rts in (missed_rally_timestamps or []):
        if isinstance(rts, datetime):
            label, _ = _period_label(rts, period)
            rally_counts[label] = rally_counts.get(label, 0) + 1

    out: List[Dict[str, Any]] = []
    for label in sorted(buckets):
        b = buckets[label]
        group = b["cycles"]
        n = len(group)
        wins = sum(1 for c in group if (c.get("return_percent") or 0.0) > 0.0)
        confs = [float(c["confidence_at_buy"]) for c in group
                 if c.get("confidence_at_buy") is not None]
        # nearest retrain accuracy at or before bucket start
        oos: Optional[float] = None
        for entry in (accuracy_history or []):
            t = entry.get("trained_at")
            if isinstance(t, str):
                try:
                    t = datetime.fromisoformat(t)
                except ValueError:
                    continue
            if isinstance(t, datetime) and t <= b["start"]:
                if entry.get("accuracy") is not None:
                    oos = float(entry["accuracy"])
        out.append({
            "label": label,
            "start": b["start"].isoformat(),
            "buy_count": n,
            "wins": wins,
            "losses": n - wins,
            "precision": (wins / n) if n else 0.0,
            "avg_return_percent": (
                sum(float(c.get("return_percent") or 0.0) for c in group) / n
            ) if n else 0.0,
            "missed_rallies": rally_counts.get(label, 0),
            "avg_confidence": (sum(confs) / len(confs)) if confs else 0.0,
            "oos_accuracy": oos,
        })
    return out


def find_missed_rallies_in_candles(
    candles: List[Tuple[datetime, float]],
) -> Optional[datetime]:
    """
    Given (timestamp, close) 5-min candles inside one HOLD gap, return the
    timestamp where a >0.3% rise within 12 bars began, or None.
    """
    hits = find_all_missed_rallies_in_candles(candles)
    return hits[0] if hits else None


def find_all_missed_rallies_in_candles(
    candles: List[Tuple[datetime, float]],
) -> List[datetime]:
    """Return distinct rally starts in a HOLD gap.

    A qualifying rally suppresses new starts for the same 12-bar evaluation
    horizon. This prevents a steadily rising sequence from being counted once
    for every candle while still allowing separate rally episodes later in the
    same signal-free gap.
    """
    n = len(candles)
    hits: List[datetime] = []
    next_allowed_index = 0
    for i in range(n):
        if i < next_allowed_index:
            continue
        base = candles[i][1]
        if not base:
            continue
        for j in range(i + 1, min(i + 1 + MISSED_RALLY_BARS, n)):
            rise = (candles[j][1] - base) / base * 100.0
            if rise > MISSED_RALLY_RISE_PERCENT:
                hits.append(candles[i][0])
                # Suppress the full evaluation horizon (MISSED_RALLY_BARS rows
                # from the start bar i), not just from the crossing bar j.
                # Using j caused a single monotonic rise to fire once per bar:
                # every candle j became the new start, then immediately found
                # the same continuing rise within 12 bars and counted another
                # "distinct" rally.  Advancing by the full window ensures that
                # once a rally episode starts at i, no new episode is
                # evaluated until the horizon has fully passed.
                next_allowed_index = i + MISSED_RALLY_BARS
                break
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# DB-backed pieces
# ─────────────────────────────────────────────────────────────────────────────

async def detect_missed_rallies(
    session: AsyncSession, ticker: str, window: str
) -> Dict[str, Any]:
    """
    Scan filtered BUY/SELL history for HOLD gaps (start→first BUY, and each
    SELL→next BUY) and check whether VOO 5-min price rose >0.3% within any
    12-bar span inside the gap. Each distinct rally episode counts once;
    the returned rate remains the fraction of gaps containing at least one.
    """
    since = datetime.utcnow() - _parse_window(window)
    try:
        result = await session.execute(
            select(SignalHistory)
            .where(and_(
                SignalHistory.ticker == ticker,
                SignalHistory.timestamp >= since,
                SignalHistory.signal_type.in_(["buy", "sell"]),
            ))
            .order_by(SignalHistory.timestamp)
        )
        rows = result.scalars().all()
        if not rows:
            # A signal-free window is itself one HOLD gap.  Returning zero
            # here hides the most important failure mode: the market rallied
            # while the signal pipeline produced nothing at all.
            gaps: List[Tuple[datetime, Optional[datetime]]] = [(since, None)]
        else:
            # Build HOLD gaps: [window start → first buy], [each sell → next buy]
            gaps = []
            gap_open: Optional[datetime] = since
            for r in rows:
                if r.signal_type == "buy":
                    if gap_open is not None:
                        gaps.append((gap_open, r.timestamp))
                        gap_open = None
                elif r.signal_type == "sell":
                    if gap_open is None:
                        gap_open = r.timestamp
            if gap_open is not None:
                gaps.append((gap_open, None))

        timestamps: List[datetime] = []
        gaps_with_rallies = 0
        for start, end in gaps:
            conds = [
                VooCandle.ticker == ticker,
                VooCandle.timeframe == "5min",
                VooCandle.timestamp >= start,
            ]
            if end is not None:
                conds.append(VooCandle.timestamp < end)
            result = await session.execute(
                select(VooCandle.timestamp, VooCandle.close)
                .where(and_(*conds))
                .order_by(VooCandle.timestamp)
            )
            candles = [(ts, float(close)) for ts, close in result.all()
                       if close is not None]
            hits = find_all_missed_rallies_in_candles(candles)
            if hits:
                gaps_with_rallies += 1
                timestamps.extend(hits)

        gap_count = len(gaps)
        return {
            "count": len(timestamps),
            "timestamps": timestamps,
            "rate": (gaps_with_rallies / gap_count) if gap_count else 0.0,
        }
    except Exception as exc:
        logger.error("detect_missed_rallies error: %s", exc)
        return {"count": 0, "timestamps": [], "rate": 0.0}


async def load_accuracy_history(
    session: AsyncSession, ticker: str
) -> List[Dict[str, Any]]:
    """Retrain OOS accuracy history from ModelMetadata (chronological)."""
    try:
        result = await session.execute(
            select(ModelMetadata)
            .where(ModelMetadata.ticker == ticker)
            .order_by(ModelMetadata.trained_at)
        )
        return [
            {
                "model_name": r.model_name,
                "trained_at": r.trained_at.isoformat()
                if isinstance(r.trained_at, datetime) else str(r.trained_at),
                "accuracy": float(r.accuracy) if r.accuracy is not None else None,
            }
            for r in result.scalars().all()
        ]
    except Exception as exc:
        logger.error("load_accuracy_history error: %s", exc)
        return []


async def compute_recommendation_stability(
    session: AsyncSession, ticker: str, window: str
) -> float:
    """Average signal flips (buy↔sell transitions) per calendar day."""
    since = datetime.utcnow() - _parse_window(window)
    try:
        result = await session.execute(
            select(SignalHistory.timestamp, SignalHistory.signal_type)
            .where(and_(
                SignalHistory.ticker == ticker,
                SignalHistory.timestamp >= since,
                SignalHistory.signal_type.in_(["buy", "sell"]),
            ))
            .order_by(SignalHistory.timestamp)
        )
        rows = result.all()
        if not rows:
            return 0.0
        flips = 0
        last_type = None
        days = set()
        for ts, sig_type in rows:
            days.add(ts.date() if isinstance(ts, datetime) else ts)
            if last_type is not None and sig_type != last_type:
                flips += 1
            last_type = sig_type
        return flips / len(days) if days else 0.0
    except Exception as exc:
        logger.error("compute_recommendation_stability error: %s", exc)
        return 0.0


async def _vix_regime_breakdown(
    session: AsyncSession, cycles: List[Dict[str, Any]], window: str
) -> Dict[str, Any]:
    """Win rate + avg return per VIX regime at cycle buy time."""
    since = datetime.utcnow() - _parse_window(window)
    vix_series = await _load_vix_close_series(session, since=since)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in cycles:
        close = _lookup_vix_close(vix_series, c.get("buy_timestamp"))
        regime = _classify_vix_regime(close) if close is not None else "NORMAL"
        groups.setdefault(regime, []).append(c)
    out: Dict[str, Any] = {}
    for regime, group in groups.items():
        n = len(group)
        wins = sum(1 for c in group if (c.get("return_percent") or 0.0) > 0.0)
        out[regime] = {
            "count": n,
            "win_rate": (wins / n) if n else 0.0,
            "average_return_percent": (
                sum(float(c.get("return_percent") or 0.0) for c in group) / n
            ) if n else 0.0,
        }
    return out


def _session_breakdown(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in cycles:
        key = c.get("session_type_at_buy") or "unknown"
        groups.setdefault(key, []).append(c)
    out: Dict[str, Any] = {}
    for key, group in groups.items():
        n = len(group)
        wins = sum(1 for c in group if (c.get("return_percent") or 0.0) > 0.0)
        out[key] = {
            "count": n,
            "win_rate": (wins / n) if n else 0.0,
            "average_return_percent": (
                sum(float(c.get("return_percent") or 0.0) for c in group) / n
            ) if n else 0.0,
        }
    return out


def _serialize_cycle(c: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if c is None:
        return None
    out = dict(c)
    for key in ("buy_timestamp", "sell_timestamp"):
        if isinstance(out.get(key), datetime):
            out[key] = out[key].isoformat()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Conviction-tier track record
# ─────────────────────────────────────────────────────────────────────────────

def _tier_group_stats(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Win rate / average return for one tier's completed cycles.

    When the sample is below MIN_TIER_SAMPLE, win_rate and
    avg_return_percent are null so clients can't render a misleading
    percentage from a tiny sample (sufficient_sample=False flags this).
    """
    n = len(group)
    sufficient = n >= MIN_TIER_SAMPLE
    if not sufficient:
        return {
            "trade_count": n,
            "win_rate": None,
            "avg_return_percent": None,
            "sufficient_sample": False,
        }
    wins = sum(1 for c in group if (c.get("return_percent") or 0.0) > 0.0)
    returns = [float(c.get("return_percent") or 0.0) for c in group]
    return {
        "trade_count": n,
        "win_rate": wins / n,
        "avg_return_percent": sum(returns) / n,
        "sufficient_sample": True,
    }


def compute_tier_track_record(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate realized outcomes per conviction tier.

    Cycles flagged price_data_absent are excluded entirely: their
    return_percent is a 0.0 sentinel, not a real outcome, and counting
    them would corrupt both win rate and average return.
    """
    usable = [c for c in cycles if not c.get("price_data_absent")]
    excluded = len(cycles) - len(usable)

    groups: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TIER_KEYS}
    for c in usable:
        tier = c.get("conviction_tier_at_buy") or "untiered"
        if tier not in groups:
            tier = "untiered"
        groups[tier].append(c)

    return {
        "overall": _tier_group_stats(usable),
        "tiers": {k: _tier_group_stats(v) for k, v in groups.items()},
        "excluded_price_data_absent": excluded,
        "min_sample_size": MIN_TIER_SAMPLE,
    }


async def get_tier_track_record(
    session: AsyncSession,
    ticker: str = "VOO",
    window: str = "90d",
) -> Dict[str, Any]:
    """Compute the /api/tier_track_record payload.

    window: '30d', '90d', or 'all' (all time).  Never raises for empty
    data — returns safe zero/null shapes.
    """
    effective_window = "3650d" if window == "all" else window
    cycles = await generate_trade_cycles(
        session, ticker=ticker, window=effective_window
    )
    payload = compute_tier_track_record(cycles)
    payload.update({
        "ticker": ticker,
        "window": window,
        "available_windows": list(TIER_WINDOWS),
    })
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def get_model_performance(
    session: AsyncSession,
    ticker: str = "VOO",
    period: str = "day",
    window: str = "90d",
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the full /api/model_performance payload. Never raises for
    empty data — returns safe zero/empty shapes."""
    if period not in VALID_PERIODS:
        period = "day"

    cycles = await generate_trade_cycles(session, ticker=ticker, window=window)
    cycles = filter_cycles_by_confidence(cycles, confidence_min, confidence_max)

    missed = await detect_missed_rallies(session, ticker, window)
    accuracy_history = await load_accuracy_history(session, ticker)
    stability = await compute_recommendation_stability(session, ticker, window)

    periods = bucket_cycles_by_period(
        cycles, period,
        missed_rally_timestamps=missed["timestamps"],
        accuracy_history=accuracy_history,
    )
    streak = compute_streaks(cycles)
    pnl = compute_cumulative_pnl(cycles)

    n = len(cycles)
    wins = sum(1 for c in cycles if (c.get("return_percent") or 0.0) > 0.0)
    confs = [float(c["confidence_at_buy"]) for c in cycles
             if c.get("confidence_at_buy") is not None]
    returns = [float(c.get("return_percent") or 0.0) for c in cycles]

    best = max(cycles, key=lambda c: c.get("return_percent") or 0.0) if cycles else None
    worst = min(cycles, key=lambda c: c.get("return_percent") or 0.0) if cycles else None

    return {
        "ticker": ticker,
        "period": period,
        "window": window,
        "summary": {
            "total_trades": n,
            "wins": wins,
            "losses": n - wins,
            "buy_precision": (wins / n) if n else 0.0,
            "avg_return_percent": (sum(returns) / n) if n else 0.0,
            "missed_rally_rate": missed["rate"],
            "current_win_streak": streak["current_win"],
            "recommendation_stability": stability,
            "avg_confidence": (sum(confs) / len(confs)) if confs else 0.0,
            "cumulative_return_percent": (
                pnl[-1]["cumulative_return_percent"] if pnl else 0.0
            ),
        },
        "periods": periods,
        "confidence_buckets": compute_confidence_buckets(cycles),
        "calibration_curve": compute_calibration_curve(cycles),
        "cumulative_pnl": pnl,
        "return_distribution": _build_return_distribution(returns),
        "session_breakdown": _session_breakdown(cycles),
        "vix_regime_breakdown": await _vix_regime_breakdown(session, cycles, window),
        "best_trade": _serialize_cycle(best),
        "worst_trade": _serialize_cycle(worst),
        "streak": streak,
        "missed_rallies": {
            "count": missed["count"],
            "timestamps": [
                t.isoformat() if isinstance(t, datetime) else str(t)
                for t in missed["timestamps"]
            ],
            "rate": missed["rate"],
        },
        "accuracy_history": accuracy_history,
    }
