"""
Signal Reliability Metrics Engine
=================================
A self-contained module that:
  1. Reconstructs the strongest-confidence BUY→SELL filtered timeline.
  2. Builds and persists trade cycles (TradeCycles table).
  3. Computes reliability metrics (win rate, returns, hold times, distributions,
     and segmentations by session type, volatility class, liquidity class).

It intentionally mirrors the filtering logic in routers/predictions.py
/filtered_signal_history without modifying that endpoint, so existing signal
logic is untouched.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import SignalHistory, TradeCycles, VixCandle, VooCandle
from signal_engine.decision_filter import DecisionFilter

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Window parsing (compatible with routers/predictions.py)
# ─────────────────────────────────────────────────────────────────────────────

_WINDOW_MAP = {
    "1d": timedelta(days=1),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
    "1y": timedelta(days=365),
    "2y": timedelta(days=730),
    "5y": timedelta(days=1825),
    "10y": timedelta(days=3650),
}


# VIX regime thresholds (mirrors indicators/technical.py conventions)
_VIX_LOW_THRESHOLD = 15.0
_VIX_NORMAL_THRESHOLD = 25.0
_VIX_HIGH_THRESHOLD = 35.0

# Default reliability score when no historical data exists for a segment
_DEFAULT_RELIABILITY_SCORE = 0.5


def _parse_window(window: str) -> timedelta:
    """Parse a window string like '30d', '1y', or '10y' into a timedelta."""
    if window in _WINDOW_MAP:
        return _WINDOW_MAP[window]
    try:
        if window.endswith("d"):
            return timedelta(days=int(window[:-1]))
        if window.endswith("h"):
            return timedelta(hours=int(window[:-1]))
        if window.endswith("m"):
            return timedelta(minutes=int(window[:-1]))
        if window.endswith("y"):
            return timedelta(days=int(window[:-1]) * 365)
    except ValueError:
        pass
    return timedelta(days=30)


# ─────────────────────────────────────────────────────────────────────────────
# Trade cycle generation from filtered BUY→SELL signals
# ─────────────────────────────────────────────────────────────────────────────

async def generate_trade_cycles(
    session: AsyncSession,
    ticker: str = "VOO",
    window: str = "30d",
    persist: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate BUY→SELL trade cycles from the strongest-confidence filtered signals.

    Steps:
      1. Load all buy/sell SignalHistory rows for the requested ticker/window.
      2. Group consecutive same-type signals and keep the highest-confidence row per group.
      3. Enforce strict BUY→SELL→BUY alternation.
      4. Match each SELL to the preceding BUY to form a cycle.
      5. Compute returns, hold times, and classifications.
      6. Persist to the TradeCycles table (idempotent by cycle_id).

    Existing signal logic is NOT modified; this function re-uses the same
    algorithmic outcome as /filtered_signal_history.
    """
    since = datetime.utcnow() - _parse_window(window)

    result = await session.execute(
        select(SignalHistory)
        .where(
            and_(
                SignalHistory.ticker == ticker,
                SignalHistory.timestamp >= since,
                SignalHistory.signal_type.in_(["buy", "sell"]),
            )
        )
        .order_by(SignalHistory.timestamp)
    )
    rows = result.scalars().all()

    if not rows:
        return []

    # Step 1: group consecutive same-type signals
    groups: List[List[SignalHistory]] = []
    current_group = [rows[0]]
    for row in rows[1:]:
        if row.signal_type == current_group[-1].signal_type:
            current_group.append(row)
        else:
            groups.append(current_group)
            current_group = [row]
    groups.append(current_group)

    # Step 2: strongest combined signal per group (confidence weighted by
    # decision-layer quality). This improves cycle pairing by preferring
    # signals that survived gap/liquidity/volatility filtering.
    best_signals = [
        max(group, key=lambda r: r.confidence * _signal_quality_score(r))
        for group in groups
    ]

    # Step 3: enforce strict alternation BUY→SELL→BUY
    filtered: List[SignalHistory] = []
    last_type: Optional[str] = None
    for sig in best_signals:
        if sig.signal_type != last_type:
            filtered.append(sig)
            last_type = sig.signal_type

    # Step 4 & 5: pair BUY→SELL and build cycles
    cycles: List[Dict[str, Any]] = []
    pending_buy: Optional[SignalHistory] = None
    for sig in filtered:
        if sig.signal_type == "buy":
            pending_buy = sig
        elif sig.signal_type == "sell" and pending_buy is not None:
            cycle = await _build_cycle(session, pending_buy, sig, ticker)
            cycles.append(cycle)
            pending_buy = None

    if persist:
        await _ensure_trade_cycles_columns(session)
        await _persist_cycles(session, cycles, ticker)

    return cycles


def _signal_quality_score(signal: SignalHistory) -> float:
    """
    In-memory quality multiplier used when selecting the best signal per group.

    Mirrors the decision-filter logic so cycle pairing favors the same kinds
    of signals that the notification layer favors: avoid gap-down and thin
    liquidity; keep other signals unchanged.
    """
    try:
        gap_type = (signal.gap_type or "none").lower()
        liquidity_score = signal.liquidity_score if signal.liquidity_score is not None else 1.0
        liquidity_class = DecisionFilter.classify_liquidity(liquidity_score)

        quality = 1.0
        if gap_type == "gap_down":
            quality -= 0.15
        if liquidity_class == "low":
            quality -= 0.25
        return max(0.1, quality)
    except Exception as exc:
        logger.error("_signal_quality_score error: %s", exc)
        return 1.0


async def _build_cycle(
    session: AsyncSession, buy: SignalHistory, sell: SignalHistory, ticker: str
) -> Dict[str, Any]:
    """Build a trade-cycle dict from a matched BUY and SELL signal."""
    # Look up the actual VOO close price at the signal timestamps when possible.
    # Long gauge is daily, short gauge is 5-minute; we pick the closest candle.
    buy_price = await _lookup_price(session, ticker, buy.timestamp, buy.gauge_type)
    sell_price = await _lookup_price(session, ticker, sell.timestamp, sell.gauge_type)

    # Guard against None or zero buy_price before any arithmetic.
    # _lookup_price is typed float but callers may patch it with None in tests;
    # a zero or missing price must never propagate a ZeroDivisionError or TypeError.
    if not buy_price or not sell_price:
        return_dollars = 0.0
        return_percent = 0.0
    else:
        return_dollars = sell_price - buy_price
        return_percent = (return_dollars / buy_price) * 100.0
    hold_time_minutes = (
        (sell.timestamp - buy.timestamp).total_seconds() / 60.0
        if sell.timestamp and buy.timestamp
        else 0.0
    )

    volatility_class = _classify_volatility(abs(return_percent))
    liquidity_class = _classify_liquidity(
        buy.liquidity_score if buy.liquidity_score is not None else 1.0
    )

    return {
        "cycle_id": str(uuid.uuid4()),
        "ticker": ticker,
        "buy_timestamp": buy.timestamp,
        "sell_timestamp": sell.timestamp,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "return_percent": return_percent,
        "return_dollars": return_dollars,
        "hold_time_minutes": hold_time_minutes,
        "confidence_at_buy": buy.confidence,
        "confidence_at_sell": sell.confidence,
        "session_type_at_buy": buy.session_type,
        "liquidity_score_at_buy": buy.liquidity_score,
        "gap_type_at_buy": buy.gap_type,
        "macro_override_applied": buy.macro_override_applied,
        "volatility_class": volatility_class,
        "liquidity_class": liquidity_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Price lookup from stored VOO candles
# ─────────────────────────────────────────────────────────────────────────────

async def _lookup_price(
    session: AsyncSession, ticker: str, ts: datetime, gauge_type: str
) -> float:
    """
    Return the closest VOO close price at or before the signal timestamp.
    Long gauge uses daily candles; short gauge uses 5-minute candles.
    Falls back to a synthetic price if no candle is available.
    """
    timeframe = "daily" if gauge_type == "long" else "5min"
    result = await session.execute(
        select(VooCandle)
        .where(
            and_(
                VooCandle.ticker == ticker,
                VooCandle.timeframe == timeframe,
                VooCandle.timestamp <= ts,
            )
        )
        .order_by(VooCandle.timestamp.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if row is not None and row.close is not None:
        return float(row.close)
    # Fallback synthetic price if no candle data is available yet
    return 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Classifications
# ─────────────────────────────────────────────────────────────────────────────

def _classify_volatility(abs_return_percent: float) -> str:
    """Classify volatility based on absolute cycle return magnitude."""
    if abs_return_percent < 0.5:
        return "low"
    if abs_return_percent < 2.0:
        return "medium"
    return "high"


def _classify_liquidity(liquidity_score: float) -> str:
    """Classify liquidity based on the SignalHistory liquidity_score."""
    if liquidity_score >= 1.0:
        return "adequate"
    return "thin"


# ─────────────────────────────────────────────────────────────────────────────
# Volatility regime & VIX helpers (in-memory, VOO-only)
# ─────────────────────────────────────────────────────────────────────────────

async def _load_vix_close_series(
    session: AsyncSession, since: datetime, ticker: str = "^VIX"
) -> pd.Series:
    """
    Load the VIX close series for the requested lookback window.

    Returns a pandas Series indexed by calendar date (timezone-naive) for
    efficient as-of lookups when a cycle buy occurred at an intraday time.
    """
    try:
        result = await session.execute(
            select(VixCandle.timestamp, VixCandle.close)
            .where(
                and_(
                    VixCandle.ticker == ticker,
                    VixCandle.timestamp >= since,
                )
            )
            .order_by(VixCandle.timestamp)
        )
        rows = result.all()
        if not rows:
            return pd.Series(dtype=float)

        index = pd.to_datetime([r[0] for r in rows]).tz_localize(None).normalize()
        values = [float(r[1]) for r in rows]
        series = pd.Series(values, index=index)
        # Multiple candles may map to the same calendar date; keep the last
        # close per date. Do NOT drop duplicate values across different dates.
        return series[~series.index.duplicated(keep="last")]
    except Exception as exc:
        logger.error("reliability_load_vix_error error=%s", exc)
        return pd.Series(dtype=float)


def _classify_vix_regime(vix_close: float) -> str:
    """Classify a single VIX close into LOW, NORMAL, HIGH, or EXTREME."""
    if vix_close < _VIX_LOW_THRESHOLD:
        return "LOW"
    if vix_close < _VIX_NORMAL_THRESHOLD:
        return "NORMAL"
    if vix_close < _VIX_HIGH_THRESHOLD:
        return "HIGH"
    return "EXTREME"


def _lookup_vix_close(vix_series: pd.Series, ts: datetime) -> Optional[float]:
    """Return the most recent VIX close on or before the given timestamp."""
    try:
        if vix_series.empty:
            return None
        target = pd.Timestamp(ts).tz_localize(None).normalize()
        asof = vix_series.asof(target)
        return float(asof) if pd.notna(asof) else None
    except Exception as exc:
        logger.error("reliability_lookup_vix_error error=%s ts=%s", exc, ts)
        return None


def _compute_volatility_regime(
    cycle: Dict[str, Any], vix_close: Optional[float]
) -> str:
    """
    Infer a volatility regime for the cycle in memory.

    Uses the VIX close at cycle buy time when available, and falls back to
    the cycle's own volatility_class. This keeps the computation VOO-only and
    does not require any schema changes.
    """
    try:
        macro_override = bool(cycle.get("macro_override_applied", False))
        volatility_class = cycle.get("volatility_class", "medium")
        vix_regime = _classify_vix_regime(vix_close) if vix_close is not None else None

        if macro_override or vix_regime == "EXTREME":
            return "macro_shock"
        if vix_regime == "HIGH" or volatility_class == "high":
            return "trending"
        if vix_regime == "LOW" and volatility_class == "low":
            return "compressed"
        if volatility_class == "low":
            return "calm"
        return "trending"
    except Exception as exc:
        logger.error("reliability_compute_volatility_regime_error error=%s", exc)
        return "calm"


def _compute_cycle_cluster_id(cycle: Dict[str, Any], volatility_regime: str) -> str:
    """Cluster ID combines the inferred volatility regime with the buy gap type."""
    gap_type = cycle.get("gap_type_at_buy") or "none"
    return f"{volatility_regime}_{gap_type}"


def _compute_win_loss_regime(cycle: Dict[str, Any]) -> str:
    """
    Win/loss regime segmentation.

    Examples: high_vol_win, low_vol_win, macro_loss, medium_vol_win, etc.
    """
    try:
        ret = float(cycle.get("return_percent", 0.0))
        is_win = ret > 0.0
        volatility_class = cycle.get("volatility_class", "medium")
        macro_override = bool(cycle.get("macro_override_applied", False))

        if macro_override:
            return "macro_win" if is_win else "macro_loss"
        if volatility_class == "high":
            return "high_vol_win" if is_win else "high_vol_loss"
        if volatility_class == "low":
            return "low_vol_win" if is_win else "low_vol_loss"
        return "medium_vol_win" if is_win else "medium_vol_loss"
    except Exception as exc:
        logger.error("reliability_compute_win_loss_regime_error error=%s", exc)
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# In-memory segmentation enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_cycles_with_segmentation(
    cycles: List[Dict[str, Any]], vix_series: pd.Series
) -> List[Dict[str, Any]]:
    """
    Add VOO-only in-memory segmentation fields to each cycle dict.

    New fields:
      - volatility_regime
      - cycle_cluster_id
      - win_loss_regime
      - gap_reliability_score
      - liquidity_reliability_score
      - session_reliability_score

    No fields are persisted; these are computed on the fly for /trade_history.
    """
    if not cycles:
        return []

    try:
        # First pass: regime + cluster labels (no aggregate dependency)
        for c in cycles:
            vix_close = _lookup_vix_close(vix_series, c.get("buy_timestamp"))
            volatility_regime = _compute_volatility_regime(c, vix_close)
            c["volatility_regime"] = volatility_regime
            c["cycle_cluster_id"] = _compute_cycle_cluster_id(c, volatility_regime)
            c["win_loss_regime"] = _compute_win_loss_regime(c)

        # Second pass: per-segment reliability scores (need aggregate stats)
        _assign_segment_reliability_score(
            cycles, segment_key="gap_type_at_buy", score_key="gap_reliability_score"
        )
        _assign_segment_reliability_score(
            cycles, segment_key="liquidity_class", score_key="liquidity_reliability_score"
        )
        _assign_segment_reliability_score(
            cycles, segment_key="session_type_at_buy", score_key="session_reliability_score"
        )
    except Exception as exc:
        logger.error("reliability_enrich_cycles_error error=%s", exc)
        # Graceful fallback: ensure every cycle has the new fields with safe defaults
        for c in cycles:
            c.setdefault("volatility_regime", "calm")
            c.setdefault("cycle_cluster_id", "calm_none")
            c.setdefault("win_loss_regime", "unknown")
            c.setdefault("gap_reliability_score", _DEFAULT_RELIABILITY_SCORE)
            c.setdefault("liquidity_reliability_score", _DEFAULT_RELIABILITY_SCORE)
            c.setdefault("session_reliability_score", _DEFAULT_RELIABILITY_SCORE)

    return cycles


def _assign_segment_reliability_score(
    cycles: List[Dict[str, Any]], segment_key: str, score_key: str
) -> Dict[str, float]:
    """
    Compute per-segment win rates and assign them back to each cycle as a score.

    Efficient single-pass aggregation: O(n) for the segment grouping and O(n)
    for the assignment, regardless of history size.
    """
    try:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for c in cycles:
            value = c.get(segment_key) or "unknown"
            groups.setdefault(value, []).append(c)

        scores: Dict[str, float] = {}
        for value, group in groups.items():
            wins = sum(1 for c in group if c.get("return_percent", 0.0) > 0.0)
            scores[value] = wins / len(group) if group else _DEFAULT_RELIABILITY_SCORE

        for c in cycles:
            c[score_key] = scores.get(c.get(segment_key) or "unknown", _DEFAULT_RELIABILITY_SCORE)
        return scores
    except Exception as exc:
        logger.error("reliability_segment_score_error key=%s error=%s", score_key, exc)
        for c in cycles:
            c[score_key] = _DEFAULT_RELIABILITY_SCORE
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

async def _ensure_trade_cycles_columns(session: AsyncSession) -> None:
    """
    Idempotently add missing columns to the trade_cycles table.

    We query PRAGMA table_info first so we only issue ALTER TABLE statements
    for columns that are not already present. This keeps each migration safe
    against repeated application.
    """
    result = await session.execute(text("PRAGMA table_info(trade_cycles)"))
    existing_columns = {row[1] for row in result.all()}

    columns_to_add = [
        ("confidence_at_buy", "FLOAT"),
        ("confidence_at_sell", "FLOAT"),
        ("session_type_at_buy", "TEXT"),
        ("liquidity_score_at_buy", "FLOAT"),
    ]
    for col_name, col_type in columns_to_add:
        if col_name not in existing_columns:
            await session.execute(
                text(f"ALTER TABLE trade_cycles ADD COLUMN {col_name} {col_type}")
            )


async def _persist_cycles(
    session: AsyncSession, cycles: List[Dict[str, Any]], ticker: str
) -> None:
    """Persist cycles idempotently using cycle_id as the unique key."""
    if not cycles:
        return

    # Fetch existing cycle_ids to avoid duplicates
    cycle_ids = [c["cycle_id"] for c in cycles]
    result = await session.execute(
        select(TradeCycles.cycle_id).where(TradeCycles.cycle_id.in_(cycle_ids))
    )
    existing = {row[0] for row in result.all()}

    new_cycles = [c for c in cycles if c["cycle_id"] not in existing]
    if not new_cycles:
        return

    session.add_all([TradeCycles(**c) for c in new_cycles])
    await session.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary reliability metrics from a list of trade cycles.

    Returns:
      - win_rate
      - average_return_percent / median_return_percent
      - average_return_dollars / median_return_dollars
      - average_hold_time
      - best_trade / worst_trade
      - return_distribution (binned histogram)
      - reliability_by_volatility_class
      - reliability_by_liquidity_class
      - reliability_by_session_type
      - reliability_by_gap_type
      - reliability_by_cycle_cluster
      - reliability_by_win_loss_regime
      - average_gap_reliability_score
      - average_liquidity_reliability_score
      - average_session_reliability_score
    """
    if not cycles:
        return {
            "win_rate": 0.0,
            "average_return_percent": 0.0,
            "median_return_percent": 0.0,
            "average_return_dollars": 0.0,
            "median_return_dollars": 0.0,
            "average_hold_time": 0.0,
            "best_trade": None,
            "worst_trade": None,
            "return_distribution": [],
            "reliability_by_volatility_class": {},
            "reliability_by_liquidity_class": {},
            "reliability_by_session_type": {},
            "reliability_by_gap_type": {},
            "reliability_by_cycle_cluster": {},
            "reliability_by_win_loss_regime": {},
            "average_gap_reliability_score": _DEFAULT_RELIABILITY_SCORE,
            "average_liquidity_reliability_score": _DEFAULT_RELIABILITY_SCORE,
            "average_session_reliability_score": _DEFAULT_RELIABILITY_SCORE,
        }

    returns = [c["return_percent"] for c in cycles]
    dollars = [c["return_dollars"] for c in cycles]
    hold_times = [c["hold_time_minutes"] for c in cycles]

    wins = sum(1 for r in returns if r > 0)
    n = len(returns)

    best = max(cycles, key=lambda c: c["return_percent"])
    worst = min(cycles, key=lambda c: c["return_percent"])

    avg_gap_score = sum(c.get("gap_reliability_score", _DEFAULT_RELIABILITY_SCORE) for c in cycles) / n
    avg_liq_score = sum(c.get("liquidity_reliability_score", _DEFAULT_RELIABILITY_SCORE) for c in cycles) / n
    avg_ses_score = sum(c.get("session_reliability_score", _DEFAULT_RELIABILITY_SCORE) for c in cycles) / n

    metrics = {
        "win_rate": wins / n if n else 0.0,
        "average_return_percent": sum(returns) / n,
        "median_return_percent": median(returns),
        "average_return_dollars": sum(dollars) / n,
        "median_return_dollars": median(dollars),
        "average_hold_time": sum(hold_times) / n,
        "best_trade": best,
        "worst_trade": worst,
        "return_distribution": _build_return_distribution(returns),
        "reliability_by_volatility_class": _segment_by(
            cycles, "volatility_class"
        ),
        "reliability_by_liquidity_class": _segment_by(
            cycles, "liquidity_class"
        ),
        "reliability_by_session_type": _segment_by(
            cycles, "session_type_at_buy"
        ),
        "reliability_by_gap_type": _segment_by(
            cycles, "gap_type_at_buy"
        ),
        "reliability_by_cycle_cluster": _segment_by(
            cycles, "cycle_cluster_id"
        ),
        "reliability_by_win_loss_regime": _segment_by(
            cycles, "win_loss_regime"
        ),
        "average_gap_reliability_score": avg_gap_score,
        "average_liquidity_reliability_score": avg_liq_score,
        "average_session_reliability_score": avg_ses_score,
    }
    return metrics


def _build_return_distribution(returns: List[float]) -> List[Dict[str, Any]]:
    """
    Build a fixed-bucket histogram of cycle returns for stable UI rendering.
    Buckets: <-2%, -2% to 0%, 0% to 2%, 2% to 5%, >=5%.
    """
    if not returns:
        return []

    buckets = [
        (-1000.0, -2.0, 0),
        (-2.0, 0.0, 0),
        (0.0, 2.0, 0),
        (2.0, 5.0, 0),
        (5.0, 1000.0, 0),
    ]
    result = []
    for lo, hi, _ in buckets:
        count = sum(1 for r in returns if lo <= r < hi)
        if hi <= -2.0:
            label = "<-2%"
        elif hi == 0.0:
            label = "-2% to 0%"
        elif lo == 0.0 and hi == 2.0:
            label = "0% to 2%"
        elif lo == 2.0 and hi == 5.0:
            label = "2% to 5%"
        else:
            label = ">=5%"
        result.append({"label": label, "min": lo, "max": hi, "count": count})
    return result


def _segment_by(cycles: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    """Segment cycles by a key (e.g., 'volatility_class') and compute per-group metrics."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in cycles:
        value = c.get(key) or "unknown"
        groups.setdefault(value, []).append(c)

    result: Dict[str, Dict[str, Any]] = {}
    for value, group in groups.items():
        returns = [c["return_percent"] for c in group]
        wins = sum(1 for r in returns if r > 0)
        result[value] = {
            "count": len(group),
            "win_rate": wins / len(group) if group else 0.0,
            "average_return_percent": sum(returns) / len(group) if group else 0.0,
            "median_return_percent": median(returns) if group else 0.0,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main endpoint helper
# ─────────────────────────────────────────────────────────────────────────────

async def get_trade_history_with_metrics(
    session: AsyncSession,
    ticker: str = "VOO",
    window: str = "30d",
) -> Dict[str, Any]:
    """
    Generate/persist cycles and return them together with summary metrics.

    This is the single helper the /trade_history endpoint uses. It keeps the
    reliability logic isolated from the rest of the signal pipeline.

    In-memory VOO-only segmentation fields (cycle_cluster_id, win_loss_regime,
    gap/liquidity/session reliability scores) are added after persistence so
    the underlying TradeCycles table schema is unchanged.
    """
    cycles = await generate_trade_cycles(session, ticker=ticker, window=window)

    # Load VIX once for the window and enrich cycles in memory only.
    since = datetime.utcnow() - _parse_window(window)
    vix_series = await _load_vix_close_series(session, since=since)
    cycles = _enrich_cycles_with_segmentation(cycles, vix_series)

    metrics = compute_metrics(cycles)

    # Convert datetime objects to ISO strings for JSON serialization
    serializable_cycles = []
    for c in cycles:
        c_copy = dict(c)
        if isinstance(c_copy.get("buy_timestamp"), datetime):
            c_copy["buy_timestamp"] = c_copy["buy_timestamp"].isoformat()
        if isinstance(c_copy.get("sell_timestamp"), datetime):
            c_copy["sell_timestamp"] = c_copy["sell_timestamp"].isoformat()
        serializable_cycles.append(c_copy)

    return {
        "ticker": ticker,
        "cycles": serializable_cycles,
        "summary": metrics,
    }
