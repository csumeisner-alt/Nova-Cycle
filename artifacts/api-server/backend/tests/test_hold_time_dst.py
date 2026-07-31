"""
Test that hold_time_minutes is computed correctly when a trade spans a
US DST spring-forward boundary (2026-03-08, clocks jump from 02:00 to 03:00
in America/New_York, i.e. the hour 02:xx simply does not exist locally).

BUY  at 2026-03-08 06:45:00 UTC  (= 01:45 EST, before the DST gap)
SELL at 2026-03-08 08:15:00 UTC  (= 04:15 EDT, after the DST gap)

In America/New_York the wall-clock difference is 2h30m = 150 min.
A naïve local-time subtraction (wrong) would give 150; the correct UTC
subtraction gives 90 min.  The backend stores and subtracts UTC timestamps
inside _build_cycle, so the expected answer is 90.

These tests call the *production* _build_cycle function and mock only
_lookup_price (which hits the database) so no real DB is required.
"""

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# Production function under test
from reliability_engine import _build_cycle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# DST spring-forward pair (2026-03-08, America/New_York)
BUY_TS  = datetime(2026, 3, 8,  6, 45, 0)   # naive UTC, = 01:45 EST
SELL_TS = datetime(2026, 3, 8,  8, 15, 0)   # naive UTC, = 04:15 EDT

EXPECTED_HOLD_MINUTES = 90.0


def _make_signal(ts: datetime, signal_type: str = "buy") -> SimpleNamespace:
    """Return a minimal SignalHistory-like object."""
    return SimpleNamespace(
        timestamp=ts,
        signal_type=signal_type,
        gauge_type="short",
        confidence=0.75,
        session_type="regular",
        is_extended_hours=False,
        gap_type="none",
        liquidity_score=1.0,
        macro_override_applied=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHoldTimeDST:
    """_build_cycle must produce 90 min for a trade spanning DST spring-forward."""

    @pytest.mark.asyncio
    async def test_dst_spanning_trade_gives_90_minutes(self):
        """
        Primary guard: _build_cycle returns hold_time_minutes == 90.0 for a BUY/SELL
        pair whose UTC timestamps are 90 minutes apart, even though the same instants
        are 150 minutes apart on a local (America/New_York) clock that crosses DST.
        """
        buy  = _make_signal(BUY_TS,  "buy")
        sell = _make_signal(SELL_TS, "sell")
        mock_session = AsyncMock()

        with patch("reliability_engine._lookup_price", new=AsyncMock(return_value=100.0)):
            cycle = await _build_cycle(mock_session, buy, sell, "VOO")

        assert cycle["hold_time_minutes"] == EXPECTED_HOLD_MINUTES, (
            f"Expected {EXPECTED_HOLD_MINUTES} min but got {cycle['hold_time_minutes']} min. "
            "hold_time_minutes must be computed from UTC timestamps, not local wall-clock time."
        )

    @pytest.mark.asyncio
    async def test_hold_time_is_utc_arithmetic_not_wall_clock(self):
        """
        Confirm that _build_cycle does NOT return the local wall-clock difference (150 min).
        If this assertion fails, the computation switched to local-time subtraction.
        """
        buy  = _make_signal(BUY_TS,  "buy")
        sell = _make_signal(SELL_TS, "sell")
        mock_session = AsyncMock()

        with patch("reliability_engine._lookup_price", new=AsyncMock(return_value=100.0)):
            cycle = await _build_cycle(mock_session, buy, sell, "VOO")

        local_wall_clock_minutes = 150.0   # what a naïve DST-unaware subtraction would give
        assert cycle["hold_time_minutes"] != local_wall_clock_minutes, (
            "hold_time_minutes equals the wrong local wall-clock difference (150 min). "
            "The computation must use UTC, not local time."
        )

    @pytest.mark.asyncio
    async def test_hold_time_zero_for_same_timestamp(self):
        """Degenerate case: BUY and SELL at the exact same UTC instant → 0 minutes."""
        ts = datetime(2026, 3, 8, 7, 0, 0)
        buy  = _make_signal(ts, "buy")
        sell = _make_signal(ts, "sell")
        mock_session = AsyncMock()

        with patch("reliability_engine._lookup_price", new=AsyncMock(return_value=100.0)):
            cycle = await _build_cycle(mock_session, buy, sell, "VOO")

        assert cycle["hold_time_minutes"] == 0.0

    @pytest.mark.asyncio
    async def test_hold_time_absent_when_timestamps_none(self):
        """If either timestamp is None, hold_time_minutes must be 0.0 (not an error)."""
        buy  = _make_signal(None, "buy")
        sell = _make_signal(SELL_TS, "sell")
        mock_session = AsyncMock()

        with patch("reliability_engine._lookup_price", new=AsyncMock(return_value=100.0)):
            cycle = await _build_cycle(mock_session, buy, sell, "VOO")

        assert cycle["hold_time_minutes"] == 0.0

    def test_dst_gap_sanity_local_time_is_150_minutes(self):
        """
        Sanity check: the same instants in local time (EST→EDT) are 150 min apart,
        confirming the test timestamps genuinely straddle the DST boundary and the
        90-min UTC answer is not trivially equal to a naïve subtraction.
        """
        buy_local  = BUY_TS  - timedelta(hours=5)   # 01:45 EST
        sell_local = SELL_TS - timedelta(hours=4)    # 04:15 EDT
        local_diff = (sell_local - buy_local).total_seconds() / 60.0
        assert local_diff == 150.0
        assert local_diff != EXPECTED_HOLD_MINUTES
