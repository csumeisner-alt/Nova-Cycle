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

import uuid
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import SignalHistory, TradeCycles, VooCandle

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

    # Step 2: strongest-confidence signal per group
    best_signals = [max(group, key=lambda r: r.confidence) for group in groups]

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


async def _build_cycle(
    session: AsyncSession, buy: SignalHistory, sell: SignalHistory, ticker: str
) -> Dict[str, Any]:
    """Build a trade-cycle dict from a matched BUY and SELL signal."""
    # Look up the actual VOO close price at the signal timestamps when possible.
    # Long gauge is daily, short gauge is 5-minute; we pick the closest candle.
    buy_price = await _lookup_price(session, ticker, buy.timestamp, buy.gauge_type)
    sell_price = await _lookup_price(session, ticker, sell.timestamp, sell.gauge_type)

    return_dollars = sell_price - buy_price
    return_percent = (return_dollars / buy_price) * 100.0 if buy_price else 0.0
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
        }

    returns = [c["return_percent"] for c in cycles]
    dollars = [c["return_dollars"] for c in cycles]
    hold_times = [c["hold_time_minutes"] for c in cycles]

    wins = sum(1 for r in returns if r > 0)
    n = len(returns)

    best = max(cycles, key=lambda c: c["return_percent"])
    worst = min(cycles, key=lambda c: c["return_percent"])

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
    """
    cycles = await generate_trade_cycles(session, ticker=ticker, window=window)
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
