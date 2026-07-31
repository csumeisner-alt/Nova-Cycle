"""
Unit tests: indicators engine handles an empty or missing VIX frame gracefully.

Covers compute_all() called directly (no HTTP layer) with:
  - pd.DataFrame()                     — completely empty, no columns
  - DataFrame with columns but no rows — still .empty == True
  - DataFrame missing the 'close' col  — has rows but wrong schema

In all cases the call must return a dict without raising, and VIX-derived
keys (vix_regime, vix_latest, vix_regime_latest) must use safe defaults.
"""

import numpy as np
import pandas as pd
import pytest

from indicators.technical import TechnicalIndicators


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_daily_df(n: int = 60) -> pd.DataFrame:
    """Minimal VOO-like daily candle frame with all required columns."""
    idx = pd.bdate_range("2026-01-02", periods=n)
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n),
            "is_extended_hours": False,
        },
        index=idx,
    )


@pytest.fixture
def ti() -> TechnicalIndicators:
    return TechnicalIndicators()


@pytest.fixture
def daily_df() -> pd.DataFrame:
    return _make_daily_df(60)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeAllEmptyVix:
    """compute_all must never raise when vix_df is absent or degenerate."""

    def _assert_safe_vix_defaults(self, result: dict) -> None:
        """Helper: confirm VIX keys exist and carry safe sentinel values."""
        assert isinstance(result, dict), "compute_all must return a dict"
        assert "vix_regime" in result
        assert "vix_latest" in result
        assert "vix_regime_latest" in result

        # Safe defaults: level 20, regime NORMAL
        assert result["vix_latest"] == 20.0, (
            f"Expected vix_latest=20.0 (safe default), got {result['vix_latest']}"
        )
        assert result["vix_regime_latest"] == "NORMAL", (
            f"Expected vix_regime_latest='NORMAL', got {result['vix_regime_latest']}"
        )

    def test_completely_empty_vix_df_returns_dict(self, ti, daily_df):
        """pd.DataFrame() — no columns, no rows — must not raise."""
        result = ti.compute_all(daily_df, pd.DataFrame())
        assert isinstance(result, dict)
        assert len(result) > 0, "Result dict should contain non-VIX indicators"

    def test_completely_empty_vix_df_uses_safe_defaults(self, ti, daily_df):
        """VIX keys fall back to level=20, regime='NORMAL'."""
        result = ti.compute_all(daily_df, pd.DataFrame())
        self._assert_safe_vix_defaults(result)

    def test_empty_vix_df_with_correct_columns_uses_safe_defaults(self, ti, daily_df):
        """DataFrame with the right schema but zero rows counts as empty."""
        empty_with_cols = pd.DataFrame(columns=["close", "open", "high", "low"])
        result = ti.compute_all(daily_df, empty_with_cols)
        assert isinstance(result, dict)
        self._assert_safe_vix_defaults(result)

    def test_vix_df_missing_close_column_uses_safe_defaults(self, ti, daily_df):
        """DataFrame with rows but missing 'close' must fall back gracefully."""
        bad_vix = pd.DataFrame({"open": [20.0, 21.0]})
        result = ti.compute_all(daily_df, bad_vix)
        assert isinstance(result, dict)
        self._assert_safe_vix_defaults(result)

    def test_non_vix_indicators_are_present_despite_empty_vix(self, ti, daily_df):
        """Core indicators must still be computed even with an empty VIX frame."""
        result = ti.compute_all(daily_df, pd.DataFrame())
        for key in ("rsi", "macd", "adx", "bollinger", "atr", "latest"):
            assert key in result, f"Expected key '{key}' missing from result"

    def test_vix_regime_series_uses_normal_when_empty(self, ti, daily_df):
        """The per-row vix_regime series should default to 'NORMAL'."""
        result = ti.compute_all(daily_df, pd.DataFrame())
        regime_series = result["vix_regime"]
        assert isinstance(regime_series, pd.Series)
        assert not regime_series.empty
        assert (regime_series == "NORMAL").all(), (
            f"Expected all 'NORMAL', got unique values: {regime_series.unique()}"
        )

    def test_latest_snapshot_vix_keys_have_safe_values(self, ti, daily_df):
        """latest dict must carry the safe VIX defaults, not None or NaN."""
        result = ti.compute_all(daily_df, pd.DataFrame())
        latest = result.get("latest", {})
        assert latest.get("vix") == 20.0
        assert latest.get("vix_regime") == "NORMAL"
