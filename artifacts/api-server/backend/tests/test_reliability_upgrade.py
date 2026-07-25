"""Tests for the VOO-only reliability metrics subsystem upgrade."""

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.db import get_session
from database.models import Base, SignalHistory, VixCandle, VooCandle
from main import app
from reliability_engine import (
    _assign_segment_reliability_score,
    _classify_vix_regime,
    _compute_cycle_cluster_id,
    _compute_volatility_regime,
    _compute_win_loss_regime,
    _enrich_cycles_with_segmentation,
    _load_vix_close_series,
    _lookup_vix_close,
    compute_metrics,
    get_trade_history_with_metrics,
)


class TestPureHelpers:
    def test_classify_vix_regime(self):
        assert _classify_vix_regime(12.0) == "LOW"
        assert _classify_vix_regime(20.0) == "NORMAL"
        assert _classify_vix_regime(30.0) == "HIGH"
        assert _classify_vix_regime(40.0) == "EXTREME"

    def test_volatility_regime_macro_override(self):
        cycle = {"macro_override_applied": True, "volatility_class": "low"}
        assert _compute_volatility_regime(cycle, 12.0) == "macro_shock"

    def test_volatility_regime_extreme_vix(self):
        cycle = {"macro_override_applied": False, "volatility_class": "low"}
        assert _compute_volatility_regime(cycle, 40.0) == "macro_shock"

    def test_volatility_regime_compressed(self):
        cycle = {"macro_override_applied": False, "volatility_class": "low"}
        assert _compute_volatility_regime(cycle, 10.0) == "compressed"

    def test_volatility_regime_trending(self):
        cycle = {"macro_override_applied": False, "volatility_class": "high"}
        assert _compute_volatility_regime(cycle, 20.0) == "trending"

    def test_volatility_regime_calm(self):
        cycle = {"macro_override_applied": False, "volatility_class": "low"}
        assert _compute_volatility_regime(cycle, 20.0) == "calm"

    def test_volatility_regime_no_vix(self):
        cycle = {"macro_override_applied": False, "volatility_class": "low"}
        assert _compute_volatility_regime(cycle, None) == "calm"

    def test_cycle_cluster_id(self):
        cycle = {"gap_type_at_buy": "gap_up"}
        assert _compute_cycle_cluster_id(cycle, "macro_shock") == "macro_shock_gap_up"

    def test_win_loss_regime(self):
        assert _compute_win_loss_regime({"return_percent": 1.0, "volatility_class": "high"}) == "high_vol_win"
        assert _compute_win_loss_regime({"return_percent": -1.0, "volatility_class": "high"}) == "high_vol_loss"
        assert _compute_win_loss_regime({"return_percent": 1.0, "volatility_class": "low"}) == "low_vol_win"
        assert _compute_win_loss_regime({"return_percent": -1.0, "volatility_class": "low"}) == "low_vol_loss"
        assert _compute_win_loss_regime({
            "return_percent": -1.0, "volatility_class": "low", "macro_override_applied": True,
        }) == "macro_loss"

    def test_assign_segment_reliability_score(self):
        cycles = [
            {"gap_type_at_buy": "gap_up", "return_percent": 1.0},
            {"gap_type_at_buy": "gap_up", "return_percent": 2.0},
            {"gap_type_at_buy": "gap_down", "return_percent": -1.0},
        ]
        _assign_segment_reliability_score(cycles, "gap_type_at_buy", "gap_reliability_score")
        assert cycles[0]["gap_reliability_score"] == 1.0
        assert cycles[1]["gap_reliability_score"] == 1.0
        assert cycles[2]["gap_reliability_score"] == 0.0

    def test_enrich_cycles_with_empty_list(self):
        assert _enrich_cycles_with_segmentation([], pd.Series(dtype=float)) == []

    def test_enrich_cycles_adds_all_fields(self):
        idx = pd.bdate_range("2026-01-02", periods=20)
        vix = pd.Series(20.0, index=idx)
        cycles = [{
            "buy_timestamp": idx[10].to_pydatetime(),
            "return_percent": 1.5,
            "volatility_class": "high",
            "gap_type_at_buy": "gap_up",
            "liquidity_class": "adequate",
            "session_type_at_buy": "regular",
            "macro_override_applied": False,
        }]
        enriched = _enrich_cycles_with_segmentation(cycles, vix)
        assert len(enriched) == 1
        c = enriched[0]
        for key in [
            "volatility_regime",
            "cycle_cluster_id",
            "win_loss_regime",
            "gap_reliability_score",
            "liquidity_reliability_score",
            "session_reliability_score",
        ]:
            assert key in c, f"Missing field: {key}"

    def test_lookup_vix_close_preserves_distinct_dates_with_same_value(self):
        """Duplicate VIX closes on different days must not collapse distinct dates."""
        idx = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"])
        vix = pd.Series([20.0, 20.0, 30.0], index=idx)
        # The second date has the same close as the first; it must still be found.
        assert _lookup_vix_close(vix, idx[1]) == 20.0
        assert _lookup_vix_close(vix, idx[2]) == 30.0
        # The first two days are identical close; as-of the second day returns the
        # second day's value (which equals the first), not the third day's 30.0.
        assert _lookup_vix_close(vix, idx[1] + pd.Timedelta(hours=12)) == 20.0

    def test_volatility_regime_changes_with_vix_across_days(self):
        """A cycle on a calm VIX day and a cycle on an extreme VIX day must differ."""
        idx = pd.to_datetime(["2026-01-02", "2026-01-03"])
        # Calm day then extreme day
        vix = pd.Series([20.0, 40.0], index=idx)
        cycles = [
            {
                "buy_timestamp": idx[0].to_pydatetime(),
                "return_percent": 0.5,
                "volatility_class": "low",
                "gap_type_at_buy": "none",
                "liquidity_class": "adequate",
                "session_type_at_buy": "regular",
                "macro_override_applied": False,
            },
            {
                "buy_timestamp": idx[1].to_pydatetime(),
                "return_percent": 0.5,
                "volatility_class": "low",
                "gap_type_at_buy": "none",
                "liquidity_class": "adequate",
                "session_type_at_buy": "regular",
                "macro_override_applied": False,
            },
        ]
        enriched = _enrich_cycles_with_segmentation(cycles, vix)
        assert enriched[0]["volatility_regime"] == "calm"
        assert enriched[0]["cycle_cluster_id"] == "calm_none"
        assert enriched[1]["volatility_regime"] == "macro_shock"
        assert enriched[1]["cycle_cluster_id"] == "macro_shock_none"


class TestMetricsWithSegmentation:
    def test_compute_metrics_includes_new_summaries(self):
        cycles = [
            {
                "return_percent": 1.0,
                "return_dollars": 1.0,
                "hold_time_minutes": 10.0,
                "volatility_class": "high",
                "liquidity_class": "adequate",
                "session_type_at_buy": "regular",
                "gap_type_at_buy": "gap_up",
                "cycle_cluster_id": "trending_gap_up",
                "win_loss_regime": "high_vol_win",
                "gap_reliability_score": 1.0,
                "liquidity_reliability_score": 1.0,
                "session_reliability_score": 1.0,
            },
            {
                "return_percent": -0.5,
                "return_dollars": -0.5,
                "hold_time_minutes": 10.0,
                "volatility_class": "low",
                "liquidity_class": "thin",
                "session_type_at_buy": "pre_market",
                "gap_type_at_buy": "none",
                "cycle_cluster_id": "calm_none",
                "win_loss_regime": "low_vol_loss",
                "gap_reliability_score": 0.0,
                "liquidity_reliability_score": 0.0,
                "session_reliability_score": 0.0,
            },
        ]
        metrics = compute_metrics(cycles)
        for key in [
            "reliability_by_gap_type",
            "reliability_by_cycle_cluster",
            "reliability_by_win_loss_regime",
            "average_gap_reliability_score",
            "average_liquidity_reliability_score",
            "average_session_reliability_score",
        ]:
            assert key in metrics, f"Missing summary key: {key}"
        assert metrics["average_gap_reliability_score"] == 0.5
        assert metrics["average_liquidity_reliability_score"] == 0.5
        assert metrics["average_session_reliability_score"] == 0.5


def _daily_candles(n=60):
    idx = pd.bdate_range(end="2026-07-24", periods=n)
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0.0, 0.3, n))
    return [
        VooCandle(
            ticker="VOO", timestamp=ts.to_pydatetime(),
            open=float(close[i]) - 0.2, high=float(close[i]) + 0.2,
            low=float(close[i]) - 0.2, close=float(close[i]),
            volume=float(rng.uniform(1e6, 5e6)),
            timeframe="daily", is_extended_hours=False,
            session_type="regular", gap_percent=0.0, gap_type="none",
        )
        for i, ts in enumerate(idx)
    ]


def _vix_candles(n=60):
    idx = pd.bdate_range(end="2026-07-24", periods=n)
    return [
        VixCandle(
            ticker="^VIX", timestamp=ts.to_pydatetime(),
            open=20.0, high=21.0, low=19.0, close=20.0,
            volume=0.0, timeframe="daily",
        )
        for ts in idx
    ]


def _signals():
    """Create alternating BUY/SELL signal pairs."""
    idx = pd.bdate_range(end="2026-07-24", periods=10)
    signals = []
    for i, ts in enumerate(idx):
        sig_type = "buy" if i % 2 == 0 else "sell"
        signals.append(SignalHistory(
            ticker="VOO",
            timestamp=ts.to_pydatetime(),
            signal_type=sig_type,
            gauge_type="long",
            confidence=0.85,
            session_type="regular",
            is_extended_hours=False,
            gap_type="gap_up" if i == 0 else "none",
            liquidity_score=1.0,
            macro_override_applied=False,
        ))
    return signals


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        s.add_all(_daily_candles() + _vix_candles() + _signals())
        await s.commit()
        yield s

    await engine.dispose()


class TestTradeHistoryIntegration:
    async def test_trade_history_includes_new_fields(self, session):
        result = await get_trade_history_with_metrics(session, ticker="VOO", window="30d")
        assert "ticker" in result
        assert "cycles" in result
        assert "summary" in result

        for c in result["cycles"]:
            for key in [
                "cycle_cluster_id",
                "win_loss_regime",
                "gap_reliability_score",
                "liquidity_reliability_score",
                "session_reliability_score",
            ]:
                assert key in c, f"Missing cycle field: {key}"

        summary = result["summary"]
        for key in [
            "reliability_by_gap_type",
            "reliability_by_cycle_cluster",
            "reliability_by_win_loss_regime",
            "average_gap_reliability_score",
            "average_liquidity_reliability_score",
            "average_session_reliability_score",
        ]:
            assert key in summary, f"Missing summary field: {key}"
