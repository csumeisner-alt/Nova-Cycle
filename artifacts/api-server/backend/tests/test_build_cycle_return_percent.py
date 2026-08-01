"""
Unit tests for _build_cycle return_percent correctness near extreme buy prices.

The computation in _build_cycle is:
    return_percent = (return_dollars / buy_price) * 100.0 if buy_price else 0.0

Covered scenarios:
  - None buy_price  → 0.0 (documented sentinel; None is falsy so guard catches it)
  - zero buy_price  → 0.0 (guard catches 0.0 which is also falsy)
  - tiny positive   → finite and reasonable (not infinite / not 0.0)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from reliability_engine import _build_cycle


# ── minimal stub for SignalHistory fields used by _build_cycle ────────────────

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


_BUY_TS = datetime(2026, 1, 10, 10, 0, 0)
_SELL_TS = datetime(2026, 1, 11, 10, 0, 0)
_BUY = _FakeSignal(_BUY_TS)
_SELL = _FakeSignal(_SELL_TS)


async def _build(buy_price, sell_price) -> dict:
    """Helper: run _build_cycle with mocked _lookup_price returning given values."""
    side_effects = [buy_price, sell_price]

    async def _fake_lookup(session, ticker, ts, gauge_type):
        return side_effects.pop(0)

    with patch("reliability_engine._lookup_price", side_effect=_fake_lookup):
        return await _build_cycle(session=None, buy=_BUY, sell=_SELL, ticker="VOO")


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_return_percent_none_buy_price_is_sentinel_zero():
    """None buy_price must produce return_percent=0.0, not a crash or NaN."""
    cycle = await _build(buy_price=None, sell_price=200.0)
    assert cycle["return_percent"] == 0.0, (
        "Expected 0.0 sentinel when buy_price is None; got %r" % cycle["return_percent"]
    )


@pytest.mark.asyncio
async def test_return_percent_zero_buy_price_is_sentinel_zero():
    """Zero buy_price must produce return_percent=0.0, not ZeroDivisionError or inf."""
    cycle = await _build(buy_price=0.0, sell_price=200.0)
    assert cycle["return_percent"] == 0.0, (
        "Expected 0.0 sentinel when buy_price is 0.0; got %r" % cycle["return_percent"]
    )


@pytest.mark.asyncio
async def test_return_percent_tiny_positive_buy_price_is_finite():
    """A very small but non-zero buy_price must produce a finite, non-zero percent."""
    buy_price = 1e-6
    sell_price = 1.5e-6  # 50 % gain
    cycle = await _build(buy_price=buy_price, sell_price=sell_price)

    rp = cycle["return_percent"]
    import math

    assert math.isfinite(rp), "return_percent must be finite for tiny buy_price; got %r" % rp
    assert rp != 0.0, "return_percent must not be zero for a profitable tiny-price cycle"
    # Sanity-check the magnitude: (sell - buy) / buy * 100 = 50.0
    expected = (sell_price - buy_price) / buy_price * 100.0
    assert abs(rp - expected) < 1e-6, (
        "return_percent %r does not match expected %r" % (rp, expected)
    )


@pytest.mark.asyncio
async def test_return_percent_normal_prices_correct():
    """Regression: a normal buy/sell pair must compute the expected percentage."""
    cycle = await _build(buy_price=100.0, sell_price=105.0)
    assert abs(cycle["return_percent"] - 5.0) < 1e-9


@pytest.mark.asyncio
async def test_return_percent_loss_cycle_is_negative():
    """A sell below buy must produce a negative return_percent."""
    cycle = await _build(buy_price=200.0, sell_price=190.0)
    assert cycle["return_percent"] < 0.0
    assert abs(cycle["return_percent"] - (-5.0)) < 1e-9


# ── sell_price sentinel tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_return_percent_none_sell_price_is_sentinel_zero():
    """None sell_price must produce return_percent=0.0 and return_dollars=0.0."""
    cycle = await _build(buy_price=100.0, sell_price=None)
    assert cycle["return_percent"] == 0.0, (
        "Expected 0.0 sentinel when sell_price is None; got %r" % cycle["return_percent"]
    )
    assert cycle["return_dollars"] == 0.0, (
        "Expected 0.0 sentinel when sell_price is None; got %r" % cycle["return_dollars"]
    )


@pytest.mark.asyncio
async def test_return_dollars_none_sell_price_is_sentinel_zero():
    """None sell_price must produce return_dollars=0.0 (no TypeError from arithmetic)."""
    cycle = await _build(buy_price=150.0, sell_price=None)
    assert cycle["return_dollars"] == 0.0


@pytest.mark.asyncio
async def test_return_percent_zero_sell_price_is_sentinel_zero():
    """Zero sell_price must produce return_percent=0.0 and return_dollars=0.0."""
    cycle = await _build(buy_price=100.0, sell_price=0.0)
    assert cycle["return_percent"] == 0.0, (
        "Expected 0.0 sentinel when sell_price is 0.0; got %r" % cycle["return_percent"]
    )
    assert cycle["return_dollars"] == 0.0, (
        "Expected 0.0 sentinel when sell_price is 0.0; got %r" % cycle["return_dollars"]
    )


@pytest.mark.asyncio
async def test_volatility_class_valid_string_when_sell_price_none():
    """volatility_class must remain a valid non-empty string even when sell_price is None."""
    cycle = await _build(buy_price=100.0, sell_price=None)
    vc = cycle.get("volatility_class")
    assert isinstance(vc, str) and vc, (
        "volatility_class must be a non-empty string; got %r" % vc
    )
    assert vc in ("low", "medium", "high"), (
        "volatility_class must be one of low/medium/high; got %r" % vc
    )


@pytest.mark.asyncio
async def test_volatility_class_valid_string_when_sell_price_zero():
    """volatility_class must remain a valid non-empty string even when sell_price is 0.0."""
    cycle = await _build(buy_price=100.0, sell_price=0.0)
    vc = cycle.get("volatility_class")
    assert isinstance(vc, str) and vc, (
        "volatility_class must be a non-empty string; got %r" % vc
    )
    assert vc in ("low", "medium", "high"), (
        "volatility_class must be one of low/medium/high; got %r" % vc
    )


@pytest.mark.asyncio
async def test_win_loss_regime_via_enrichment_when_sell_price_none():
    """
    _compute_win_loss_regime must return a valid string when return_percent is the
    sentinel 0.0 (triggered by a None sell_price).  The function must not crash.
    """
    from reliability_engine import _compute_win_loss_regime

    # Simulate the cycle dict that _build_cycle produces when sell_price is None
    cycle = await _build(buy_price=100.0, sell_price=None)
    regime = _compute_win_loss_regime(cycle)
    assert isinstance(regime, str) and regime, (
        "win_loss_regime must be a non-empty string; got %r" % regime
    )
    # return_percent == 0.0 is not a win, so the regime must reflect a loss label
    assert "win" not in regime or "loss" in regime or regime == "unknown", (
        "A zero-return cycle should not be classified as a win; got %r" % regime
    )
