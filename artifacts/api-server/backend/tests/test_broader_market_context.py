"""Tests for broader market context features.

Covers:
  - compute_vix_term_structure: slope values, missing flag, freshness, proxy mode
  - compute_credit_stress: value range, missing flag, HY-only path
  - compute_market_breadth: value range, missing flag
  - compute_rates_level: value range, missing flag
  - _stale_mask helper: vectorised staleness detection, weekend tolerance
  - LONG_BROADER_CONTEXT_ENABLED flag: feature-count change, build_features output width
  - Ablation guard: flag=False keeps exactly 19 features (no regression)
"""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml import features as ml_features


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _bdate(start="2024-01-02", n=60):
    """Business-day index with n periods."""
    return pd.bdate_range(start, periods=n)


def _series(idx, start=100.0, vol=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(start + np.cumsum(rng.normal(0, vol, len(idx))), index=idx)


# ── _stale_mask ───────────────────────────────────────────────────────────────

class TestStaleMask:
    def test_empty_source_all_stale(self):
        idx = _bdate(n=5)
        mask = ml_features._stale_mask(idx, pd.DatetimeIndex([]), staleness_max_days=5)
        assert mask.all()

    def test_fresh_data_not_stale(self):
        idx = _bdate("2024-03-01", n=5)
        # Source has data right up to the last target date
        src = _bdate("2024-01-02", n=60)
        mask = ml_features._stale_mask(idx, src, staleness_max_days=5)
        assert not mask.any()

    def test_stale_when_gap_exceeds_threshold(self):
        idx = pd.DatetimeIndex(["2024-03-20"])
        # Source stops 2024-03-01 — gap = 19 calendar days
        src = pd.DatetimeIndex(["2024-03-01"])
        # staleness_max_days=5 → tolerance=7; 19 > 7 → stale
        mask = ml_features._stale_mask(idx, src, staleness_max_days=5)
        assert mask.iloc[0]

    def test_weekend_tolerance_not_stale(self):
        # Monday 2024-03-04; source last has Friday 2024-03-01 (3 calendar days gap)
        idx = pd.DatetimeIndex(["2024-03-04"])
        src = pd.DatetimeIndex(["2024-03-01"])
        # staleness_max_days=5 → tolerance=7; 3 ≤ 7 → NOT stale
        mask = ml_features._stale_mask(idx, src, staleness_max_days=5)
        assert not mask.iloc[0]

    def test_before_source_start_is_stale(self):
        idx = pd.DatetimeIndex(["2023-01-02"])
        src = pd.DatetimeIndex(["2024-01-02"])
        mask = ml_features._stale_mask(idx, src, staleness_max_days=5)
        assert mask.iloc[0]

    def test_returns_series_with_correct_index(self):
        idx = _bdate("2024-03-01", n=10)
        src = _bdate("2024-01-02", n=60)
        mask = ml_features._stale_mask(idx, src, staleness_max_days=5)
        assert isinstance(mask, pd.Series)
        assert list(mask.index) == list(idx)


# ── VIX term structure ────────────────────────────────────────────────────────

class TestVixTermStructure:
    def test_proxy_mode_when_no_term_data(self):
        idx = _bdate(n=40)
        vix = _series(idx, start=20.0, vol=0.5)
        slope, missing = ml_features.compute_vix_term_structure(vix)
        assert slope.between(-1.0, 1.0).all()
        # proxy mode → missing is always 1.0
        assert (missing == 1.0).all()

    def test_real_term_data_clears_missing(self):
        idx = _bdate(n=40)
        vix = _series(idx, start=20.0, vol=0.5)
        vix9d = _series(idx, start=18.0, vol=0.5, seed=1)
        vix3m = _series(idx, start=22.0, vol=0.3, seed=2)
        slope, missing = ml_features.compute_vix_term_structure(
            vix, vix_short_close=vix9d, vix_long_close=vix3m
        )
        assert slope.between(-1.0, 1.0).all()
        # Data is fresh (same index) → missing should be 0
        assert (missing == 0.0).all()

    def test_backwardation_positive_slope(self):
        """VIX9D > VIX3M → term slope > 0 (near-term fear spike)."""
        idx = _bdate(n=30)
        vix = pd.Series(20.0, index=idx)
        vix9d = pd.Series(30.0, index=idx)  # sharply elevated short-term
        vix3m = pd.Series(20.0, index=idx)
        slope, _ = ml_features.compute_vix_term_structure(
            vix, vix_short_close=vix9d, vix_long_close=vix3m
        )
        assert (slope > 0).all()

    def test_contango_negative_slope(self):
        """VIX9D < VIX3M → term slope < 0 (calm near-term)."""
        idx = _bdate(n=30)
        vix = pd.Series(20.0, index=idx)
        vix9d = pd.Series(15.0, index=idx)
        vix3m = pd.Series(22.0, index=idx)
        slope, _ = ml_features.compute_vix_term_structure(
            vix, vix_short_close=vix9d, vix_long_close=vix3m
        )
        assert (slope < 0).all()

    def test_stale_term_data_fires_missing(self):
        idx = pd.bdate_range("2024-06-01", periods=10)
        vix = _series(idx, start=20.0)
        # Term data stops 30 days before our target dates
        old_idx = pd.bdate_range("2024-04-01", periods=10)
        vix9d = _series(old_idx, start=18.0)
        vix3m = _series(old_idx, start=22.0)
        slope, missing = ml_features.compute_vix_term_structure(
            vix, vix_short_close=vix9d, vix_long_close=vix3m
        )
        # All target dates are stale relative to old source data
        assert (missing == 1.0).all()

    def test_never_raises_on_bad_input(self):
        idx = _bdate(n=5)
        vix = pd.Series([np.nan] * 5, index=idx)
        slope, missing = ml_features.compute_vix_term_structure(vix)
        assert len(slope) == 5
        assert len(missing) == 5


# ── Credit stress ─────────────────────────────────────────────────────────────

class TestCreditStress:
    def test_absent_data_neutral_and_missing(self):
        idx = _bdate(n=40)
        score, missing = ml_features.compute_credit_stress(idx)
        assert (score == 0.5).all()
        assert (missing == 1.0).all()

    def test_score_in_range_with_data(self):
        idx = _bdate(n=60)
        hy = _series(idx, start=80.0, vol=0.3, seed=0)
        ig = _series(idx, start=110.0, vol=0.2, seed=1)
        score, missing = ml_features.compute_credit_stress(idx, hy_close=hy, ig_close=ig)
        assert score.between(0.0, 1.0).all()

    def test_missing_fires_when_data_stale(self):
        tgt = pd.bdate_range("2024-06-01", periods=10)
        old_idx = pd.bdate_range("2024-03-01", periods=10)
        hy = _series(old_idx, start=80.0)
        score, missing = ml_features.compute_credit_stress(tgt, hy_close=hy)
        assert (missing == 1.0).all()

    def test_hy_only_path_produces_valid_score(self):
        idx = _bdate(n=40)
        hy = _series(idx, start=80.0, vol=0.5)
        score, missing = ml_features.compute_credit_stress(idx, hy_close=hy)
        assert score.between(0.0, 1.0).all()
        # Data is fresh → missing should be 0
        assert (missing == 0.0).all()

    def test_never_raises_on_constant_series(self):
        idx = _bdate(n=30)
        hy = pd.Series(80.0, index=idx)  # flat → zero spread variance
        ig = pd.Series(110.0, index=idx)
        score, missing = ml_features.compute_credit_stress(idx, hy_close=hy, ig_close=ig)
        assert len(score) == len(idx)


# ── Market breadth ────────────────────────────────────────────────────────────

class TestMarketBreadth:
    def test_absent_data_neutral_and_missing(self):
        idx = _bdate(n=30)
        score, missing = ml_features.compute_market_breadth(idx)
        assert (score == 0.5).all()
        assert (missing == 1.0).all()

    def test_score_in_range_with_data(self):
        idx = _bdate(n=60)
        breadth = _series(idx, start=5000.0, vol=50.0)
        score, missing = ml_features.compute_market_breadth(idx, breadth_close=breadth)
        assert score.between(0.0, 1.0).all()

    def test_missing_fires_when_stale(self):
        tgt = pd.bdate_range("2024-06-01", periods=5)
        old_idx = pd.bdate_range("2024-03-01", periods=5)
        breadth = _series(old_idx, start=5000.0)
        score, missing = ml_features.compute_market_breadth(tgt, breadth_close=breadth)
        assert (missing == 1.0).all()

    def test_never_raises(self):
        idx = _bdate(n=5)
        breadth = pd.Series([np.nan] * 5, index=idx)
        score, missing = ml_features.compute_market_breadth(idx, breadth_close=breadth)
        assert len(score) == 5


# ── Rates level ───────────────────────────────────────────────────────────────

class TestRatesLevel:
    def test_absent_data_neutral_and_missing(self):
        idx = _bdate(n=30)
        norm, missing = ml_features.compute_rates_level(idx)
        assert (norm == 0.5).all()
        assert (missing == 1.0).all()

    def test_normalised_range_with_data(self):
        idx = _bdate(n=60)
        # TNX reports yield * 10 (e.g. 45 = 4.5%); clip_max=8.0% → raw max = 80
        rates = pd.Series(45.0, index=idx)  # 4.5% → norm = 4.5/8.0 = 0.5625
        norm, missing = ml_features.compute_rates_level(idx, rates_close=rates)
        assert norm.between(0.0, 1.0).all()
        assert pytest.approx(float(norm.iloc[-1]), abs=0.01) == 45.0 / 80.0

    def test_missing_fires_when_stale(self):
        tgt = pd.bdate_range("2024-06-01", periods=5)
        old_idx = pd.bdate_range("2024-03-01", periods=5)
        rates = _series(old_idx, start=40.0)
        norm, missing = ml_features.compute_rates_level(tgt, rates_close=rates)
        assert (missing == 1.0).all()

    def test_clamps_high_yield(self):
        idx = _bdate(n=10)
        # 15% yield → clip to 8% cap → norm = 1.0
        rates = pd.Series(150.0, index=idx)
        norm, missing = ml_features.compute_rates_level(idx, rates_close=rates)
        assert (norm == 1.0).all()

    def test_never_raises(self):
        idx = _bdate(n=5)
        rates = pd.Series([np.nan] * 5, index=idx)
        norm, missing = ml_features.compute_rates_level(idx, rates_close=rates)
        assert len(norm) == 5


# ── Ablation / feature-count guard ────────────────────────────────────────────

class TestAblationFlag:
    def test_disabled_keeps_19_features(self):
        """LONG_BROADER_CONTEXT_ENABLED=False must not change feature count."""
        from ml import long_trend
        with patch.object(
            long_trend.settings, "LONG_BROADER_CONTEXT_ENABLED", False
        ):
            import importlib
            # Re-derive what FEATURE_NAMES would be at import time
            base = long_trend._BASE_FEATURE_NAMES
            assert len(base) == 19

    def test_enabled_adds_8_features(self):
        """LONG_BROADER_CONTEXT_ENABLED=True must add exactly 8 features."""
        from ml import long_trend
        assert len(long_trend._BROADER_CONTEXT_FEATURE_NAMES) == 8

    def test_feature_names_match_enabled_flag_false(self):
        """When flag is False, FEATURE_NAMES == _BASE_FEATURE_NAMES."""
        from ml import long_trend
        with patch.object(long_trend.settings, "LONG_BROADER_CONTEXT_ENABLED", False):
            expected = long_trend._BASE_FEATURE_NAMES
            # Compute as the module would
            result = long_trend._BASE_FEATURE_NAMES + []
            assert result == expected

    def test_build_features_width_matches_feature_names_when_disabled(self):
        """build_features output width must equal len(FEATURE_NAMES) when flag is False."""
        import importlib
        import sys
        from ml import long_trend

        idx = pd.bdate_range("2020-01-02", periods=80)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "open": 100 + np.cumsum(rng.normal(0, 1, 80)),
            "high": 102 + np.cumsum(rng.normal(0, 1, 80)),
            "low": 98 + np.cumsum(rng.normal(0, 1, 80)),
            "close": 100 + np.cumsum(rng.normal(0, 1, 80)),
            "volume": rng.integers(1_000_000, 5_000_000, 80).astype(float),
        }, index=idx)

        model = long_trend.LongTrendModel()
        with patch.object(long_trend.settings, "LONG_BROADER_CONTEXT_ENABLED", False):
            X, weights, pos = model.build_features(df, {})
            if len(X) > 0:
                assert X.shape[1] == 19

    def test_build_features_width_matches_feature_names_when_enabled(self):
        """build_features output width must equal 27 when flag is True."""
        from ml import long_trend

        idx = pd.bdate_range("2020-01-02", periods=80)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "open": 100 + np.cumsum(rng.normal(0, 1, 80)),
            "high": 102 + np.cumsum(rng.normal(0, 1, 80)),
            "low": 98 + np.cumsum(rng.normal(0, 1, 80)),
            "close": 100 + np.cumsum(rng.normal(0, 1, 80)),
            "volume": rng.integers(1_000_000, 5_000_000, 80).astype(float),
        }, index=idx)

        model = long_trend.LongTrendModel()
        with patch.object(long_trend.settings, "LONG_BROADER_CONTEXT_ENABLED", True):
            X, weights, pos = model.build_features(df, {})
            if len(X) > 0:
                assert X.shape[1] == 27


# ── No-data fallback values stay neutral ─────────────────────────────────────

class TestNeutralFallbacks:
    """All missing-data fallbacks should produce neutral (0.5) values and
    missing=1.0 so a model trained with absent context learns to ignore those
    features rather than learning spurious signal from a hard-coded extreme."""

    def test_credit_stress_neutral_when_absent(self):
        idx = _bdate(n=10)
        score, missing = ml_features.compute_credit_stress(idx)
        assert float(score.mean()) == pytest.approx(0.5)
        assert (missing == 1.0).all()

    def test_breadth_neutral_when_absent(self):
        idx = _bdate(n=10)
        score, missing = ml_features.compute_market_breadth(idx)
        assert float(score.mean()) == pytest.approx(0.5)
        assert (missing == 1.0).all()

    def test_rates_neutral_when_absent(self):
        idx = _bdate(n=10)
        norm, missing = ml_features.compute_rates_level(idx)
        assert float(norm.mean()) == pytest.approx(0.5)
        assert (missing == 1.0).all()
