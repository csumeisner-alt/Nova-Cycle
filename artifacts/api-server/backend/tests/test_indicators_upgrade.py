"""Tests for the VOO-only indicator subsystem upgrade."""

import numpy as np
import pandas as pd
import pytest

from indicators.technical import TechnicalIndicators


def _daily_df(n=120, gap=0.0):
    idx = pd.bdate_range("2026-01-02", periods=n)
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)) + gap, index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(100.0),
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
        "is_extended_hours": False,
    }, index=idx)


@pytest.fixture
def ti():
    return TechnicalIndicators()


class TestVolatilityRegimeHelper:
    def test_returns_known_labels(self, ti):
        df = _daily_df(120)
        regime = ti._compute_volatility_regime(df["close"])
        assert set(regime.unique()) <= {"calm", "trending", "compressed", "macro_shock"}
        assert len(regime) == len(df)

    def test_extreme_vix_forces_macro_shock(self, ti):
        df = _daily_df(60)
        vix = pd.Series("EXTREME", index=df.index)
        regime = ti._compute_volatility_regime(df["close"], vix_regime=vix)
        assert (regime == "macro_shock").all()

    def test_empty_inputs_return_calm(self, ti):
        empty = pd.Series(dtype=float)
        regime = ti._compute_volatility_regime(empty)
        assert len(regime) == 0


class TestAdaptiveRsi:
    def test_falls_back_without_regime(self, ti):
        df = _daily_df(60)
        rsi = ti.compute_adaptive_rsi(df["close"])
        assert len(rsi) == len(df)
        assert ((rsi >= 0) & (rsi <= 100)).all()

    def test_regime_changes_period(self, ti):
        df = _daily_df(60)
        regime = pd.Series(["macro_shock"] * 30 + ["calm"] * 30, index=df.index)
        rsi = ti.compute_adaptive_rsi(df["close"], regime)
        # The macro-shock portion uses a shorter period and should react faster
        # than the calm portion. We just verify it returns valid values.
        assert ((rsi >= 0) & (rsi <= 100)).all()
        assert rsi.notna().all()

    def test_error_fallback(self, ti):
        bad = pd.Series([1, 2], index=pd.date_range("2026-01-01", periods=2))
        # Passing an empty volatility regime triggers the standard RSI fallback.
        rsi = ti.compute_adaptive_rsi(bad, volatility_regime=pd.Series())
        assert ((rsi >= 0) & (rsi <= 100)).all()


class TestEmaRibbon:
    def test_bullish_alignment(self, ti):
        # Construct a perfectly rising price series → shorter EMAs above longer EMAs
        idx = pd.bdate_range("2026-01-02", periods=60)
        close = pd.Series(100 + np.linspace(0, 50, 60), index=idx)
        ribbon = ti.compute_ema_ribbon(close)
        assert "alignment" in ribbon
        assert ribbon["alignment"].iloc[-1] == "bullish"
        for p in [8, 13, 21, 34, 55]:
            assert f"ema{p}" in ribbon

    def test_bearish_alignment(self, ti):
        idx = pd.bdate_range("2026-01-02", periods=60)
        close = pd.Series(150 - np.linspace(0, 50, 60), index=idx)
        ribbon = ti.compute_ema_ribbon(close)
        assert ribbon["alignment"].iloc[-1] == "bearish"

    def test_error_fallback(self, ti):
        ribbon = ti.compute_ema_ribbon(pd.Series(dtype=float))
        assert ribbon["alignment"].empty


class TestAtrCompression:
    def test_range(self, ti):
        df = _daily_df(120)
        score = ti.compute_atr_compression_score(df["high"], df["low"], df["close"])
        assert ((score >= 0.0) & (score <= 1.0)).all()
        assert len(score) == len(df)

    def test_low_volatility_is_compressed(self, ti):
        # Volatile history followed by a flat low-range squeeze → compression near 1
        idx = pd.bdate_range("2026-01-02", periods=120)
        rng = np.random.default_rng(7)
        noisy = 100 + np.cumsum(rng.normal(0, 1.0, 100))
        flat = np.full(20, noisy[-1])
        close = pd.Series(np.concatenate([noisy, flat]), index=idx)
        high = close + pd.Series([0.5] * 100 + [0.01] * 20, index=idx)
        low = close - pd.Series([0.5] * 100 + [0.01] * 20, index=idx)
        score = ti.compute_atr_compression_score(high, low, close)
        assert score.iloc[-1] > 0.7


class TestBollingerSlope:
    def test_reuses_existing_bollinger_values(self, ti):
        close = _daily_df(60)["close"]
        bollinger = ti.compute_bollinger_bands(close)
        slope = ti.compute_bollinger_slope(close, bollinger_data=bollinger)
        assert len(slope) == len(close)
        assert np.isfinite(slope).all()

    def test_flat_middle_band_zero_slope(self, ti):
        close = pd.Series(100.0, index=pd.bdate_range("2026-01-02", periods=40))
        slope = ti.compute_bollinger_slope(close)
        assert (slope == 0.0).all()


class TestTrendStrengthIndex:
    def test_range(self, ti):
        idx = pd.bdate_range("2026-01-02", periods=60)
        alignment = pd.Series(["bullish", "neutral", "bearish"] * 20, index=idx)
        rsi = pd.Series(50.0, index=idx)
        atr = pd.Series(0.3, index=idx)
        tsi = ti.compute_trend_strength_index(alignment, rsi, atr)
        assert ((tsi >= 0.0) & (tsi <= 1.0)).all()
        assert len(tsi) == len(idx)

    def test_bullish_extreme_rsi_high(self, ti):
        idx = pd.bdate_range("2026-01-02", periods=60)
        alignment = pd.Series("bullish", index=idx)
        rsi = pd.Series(90.0, index=idx)
        atr = pd.Series(0.0, index=idx)  # no compression
        tsi = ti.compute_trend_strength_index(alignment, rsi, atr)
        assert tsi.iloc[-1] > 0.75


class TestComputeAllIntegration:
    def test_new_indicators_present(self, ti):
        df = _daily_df(120)
        vix_df = pd.DataFrame({"close": pd.Series(20.0, index=df.index)})
        indicators = ti.compute_all(df, vix_df, exclude_extended=True)

        for key in [
            "adaptive_rsi",
            "ema_ribbon",
            "atr_compression_score",
            "bollinger_slope",
            "trend_strength_index",
        ]:
            assert key in indicators, f"Missing indicator: {key}"

        latest = indicators.get("latest", {})
        for key in [
            "adaptive_rsi",
            "ema_ribbon_alignment",
            "atr_compression_score",
            "bollinger_slope",
            "trend_strength_index",
        ]:
            assert key in latest, f"Missing latest snapshot key: {key}"

        assert latest["ema_ribbon_alignment"] in {"bullish", "neutral", "bearish"}
        assert 0.0 <= latest["trend_strength_index"] <= 1.0

    def test_legacy_latest_keys_preserved(self, ti):
        """compute_all must keep all existing indicator snapshot keys intact."""
        df = _daily_df(120)
        vix_df = pd.DataFrame({"close": pd.Series(20.0, index=df.index)})
        indicators = ti.compute_all(df, vix_df, exclude_extended=True)
        latest = indicators.get("latest", {})

        expected_legacy_keys = {
            "close", "sma50", "sma200", "sma20", "macd", "macd_signal",
            "macd_histogram", "adx", "atr", "rsi", "stoch_k", "stoch_d",
            "stoch_rsi_k", "stoch_rsi_d", "bb_upper", "bb_lower",
            "bb_pct_b", "bb_bandwidth", "cci", "williams_r", "vix",
            "vix_regime",
        }
        missing = expected_legacy_keys - set(latest.keys())
        assert not missing, f"Legacy latest keys missing: {missing}"
