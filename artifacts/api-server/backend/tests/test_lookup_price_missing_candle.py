"""
Unit tests for _lookup_price returning None when no candle data is available.

Verifies that:
  - _lookup_price returns None (not a synthetic 100.0) when the DB has no matching row
  - _lookup_price returns None when the matching row has a NULL close
  - _build_cycle, given that None, records return_percent=0.0 as the documented sentinel
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliability_engine import _build_cycle, _lookup_price


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_session(row=None):
    """Return a minimal AsyncSession stub whose execute() returns the given row."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = row

    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    return session


class _FakeSignal:
    """Lightweight stand-in for SignalHistory – only fields _build_cycle reads."""

    def __init__(self, ts: datetime, gauge_type: str = "long"):
        self.timestamp = ts
        self.gauge_type = gauge_type
        self.confidence = 0.85
        self.session_type = "regular"
        self.liquidity_score = 1.0
        self.gap_type = "none"
        self.macro_override_applied = False


_BUY_TS = datetime(2026, 3, 1, 9, 30, 0)
_SELL_TS = datetime(2026, 3, 2, 9, 30, 0)


# ── _lookup_price direct tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lookup_price_returns_none_when_no_candle_row():
    """_lookup_price must return None, not 100.0, when the DB has no matching candle."""
    session = _make_session(row=None)
    price = await _lookup_price(session, "VOO", _BUY_TS, "long")
    assert price is None, (
        "Expected None when no candle row exists; got %r — "
        "a synthetic $100 fallback would silently corrupt return_percent" % price
    )


@pytest.mark.asyncio
async def test_lookup_price_returns_none_when_candle_has_null_close():
    """_lookup_price must return None when the candle row's close field is NULL."""
    row = MagicMock()
    row.close = None
    session = _make_session(row=row)
    price = await _lookup_price(session, "VOO", _BUY_TS, "long")
    assert price is None, (
        "Expected None when candle.close is None; got %r" % price
    )


@pytest.mark.asyncio
async def test_lookup_price_returns_real_close_when_candle_present():
    """Sanity check: _lookup_price must return the actual close when a row exists."""
    row = MagicMock()
    row.close = 487.32
    session = _make_session(row=row)
    price = await _lookup_price(session, "VOO", _BUY_TS, "long")
    assert price == pytest.approx(487.32), (
        "Expected 487.32 from candle.close; got %r" % price
    )


# ── end-to-end: missing candle → cycle sentinel ───────────────────────────────

@pytest.mark.asyncio
async def test_build_cycle_flags_missing_candle_with_sentinel_return_percent():
    """
    End-to-end: when _lookup_price returns None for both buy and sell (no candle data),
    _build_cycle must record return_percent=0.0 as the documented sentinel rather than
    computing a return from a synthetic $100 price.
    """
    buy = _FakeSignal(_BUY_TS)
    sell = _FakeSignal(_SELL_TS)

    # _lookup_price returns None for every call (no candle available)
    async def _no_candle(session, ticker, ts, gauge_type):
        return None

    with patch("reliability_engine._lookup_price", side_effect=_no_candle):
        cycle = await _build_cycle(session=None, buy=buy, sell=sell, ticker="VOO")

    assert cycle["return_percent"] == 0.0, (
        "Cycle built from missing candle data must have return_percent=0.0 sentinel; "
        "got %r — the old 100.0 fallback would have produced a non-zero, misleading value"
        % cycle["return_percent"]
    )
    assert cycle["return_dollars"] == 0.0, (
        "Cycle built from missing candle data must have return_dollars=0.0; got %r"
        % cycle["return_dollars"]
    )


@pytest.mark.asyncio
async def test_build_cycle_missing_buy_candle_only_is_sentinel():
    """
    When only the buy price is missing (sell price exists), the cycle must still
    record return_percent=0.0 — a return cannot be computed without a buy price.
    """
    buy = _FakeSignal(_BUY_TS)
    sell = _FakeSignal(_SELL_TS)

    prices = [None, 492.10]  # buy missing, sell present

    async def _lookup(session, ticker, ts, gauge_type):
        return prices.pop(0)

    with patch("reliability_engine._lookup_price", side_effect=_lookup):
        cycle = await _build_cycle(session=None, buy=buy, sell=sell, ticker="VOO")

    assert cycle["return_percent"] == 0.0, (
        "Missing buy price alone must produce return_percent=0.0; got %r"
        % cycle["return_percent"]
    )
