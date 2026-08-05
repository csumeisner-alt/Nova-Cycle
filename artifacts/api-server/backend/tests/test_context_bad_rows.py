"""Tests: bad context candle rows can't corrupt live predictions.

Covers:
  - _sanitize_close: NaN detection, zero-close detection, negative-close
    detection, logging, fill strategy, all-bad series safety
  - compute_vix_term_structure: NaN / zero close fires WARNING and sets
    vix_term_missing=1.0 at affected positions
  - compute_credit_stress: NaN / zero close for HYG or LQD fires WARNING
    and sets credit_stress_missing=1.0 at affected positions
  - compute_market_breadth: NaN / zero close for NYAD fires WARNING and
    sets breadth_missing=1.0 at affected positions
  - compute_rates_level: NaN / zero close for TNX fires WARNING and sets
    rates_missing=1.0 at affected positions
  - Invariant: no bad value passes raw to the model — score/slope stays
    within expected range even when the input is corrupt
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ml import features as ml_features


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bdate(start="2024-01-02", n=40):
    return pd.bdate_range(start, periods=n)


def _clean(idx, start=100.0, vol=1.0, seed=0):
    """Well-behaved positive series."""
    rng = np.random.default_rng(seed)
    return pd.Series(
        start + np.cumsum(rng.normal(0, vol, len(idx))),
        index=idx,
    )


def _inject_bad(series: pd.Series, positions: list[int], value) -> pd.Series:
    """Return a copy of *series* with *value* at the given integer positions."""
    s = series.copy()
    for pos in positions:
        s.iloc[pos] = value
    return s


# ─────────────────────────────────────────────────────────────────────────────
# _sanitize_close
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizeClose:
    def test_clean_series_no_bad_mask(self):
        idx = _bdate(n=10)
        s = _clean(idx, start=50.0)
        cleaned, bad_mask = ml_features._sanitize_close(s, "test")
        assert not bad_mask.any()
        pd.testing.assert_series_equal(cleaned, s.astype(float), check_names=False)

    def test_nan_detected(self):
        idx = _bdate(n=10)
        s = _inject_bad(_clean(idx, start=50.0), [3], np.nan)
        _, bad_mask = ml_features._sanitize_close(s, "test")
        assert bad_mask.iloc[3]
        assert bad_mask.sum() == 1

    def test_zero_close_detected(self):
        idx = _bdate(n=10)
        s = _inject_bad(_clean(idx, start=50.0), [5], 0.0)
        _, bad_mask = ml_features._sanitize_close(s, "test")
        assert bad_mask.iloc[5]

    def test_negative_close_detected(self):
        idx = _bdate(n=10)
        s = _inject_bad(_clean(idx, start=50.0), [2], -1.5)
        _, bad_mask = ml_features._sanitize_close(s, "test")
        assert bad_mask.iloc[2]

    def test_inf_detected(self):
        idx = _bdate(n=10)
        s = _inject_bad(_clean(idx, start=50.0), [7], np.inf)
        _, bad_mask = ml_features._sanitize_close(s, "test")
        assert bad_mask.iloc[7]

    def test_cleaned_values_are_positive(self):
        """Bad positions must be filled so all values in cleaned are > 0."""
        idx = _bdate(n=10)
        s = _inject_bad(_clean(idx, start=50.0), [0, 5, 9], np.nan)
        cleaned, _ = ml_features._sanitize_close(s, "test")
        assert (cleaned > 0).all()

    def test_all_bad_series_stays_safe(self):
        """An entirely-bad series must not raise and must return a numeric series."""
        idx = _bdate(n=5)
        s = pd.Series([0.0, np.nan, -1.0, 0.0, np.nan], index=idx)
        cleaned, bad_mask = ml_features._sanitize_close(s, "all_bad")
        assert bad_mask.all()
        assert cleaned.notna().all()
        assert (cleaned > 0).all()

    def test_warning_logged_for_bad_rows(self, caplog):
        idx = _bdate(n=10)
        s = _inject_bad(_clean(idx, start=50.0), [2, 6], 0.0)
        with caplog.at_level(logging.WARNING, logger="ml.features"):
            ml_features._sanitize_close(s, "hyg_close")
        assert any(
            "ml_feature_bad_context_row" in r.message
            and "hyg_close" in r.message
            for r in caplog.records
        )

    def test_no_warning_for_clean_series(self, caplog):
        idx = _bdate(n=10)
        s = _clean(idx, start=50.0)
        with caplog.at_level(logging.WARNING, logger="ml.features"):
            ml_features._sanitize_close(s, "clean_label")
        assert not any(
            "ml_feature_bad_context_row" in r.message for r in caplog.records
        )

    def test_warning_includes_bad_count(self, caplog):
        idx = _bdate(n=20)
        s = _inject_bad(_clean(idx, start=50.0), [1, 5, 10], np.nan)
        with caplog.at_level(logging.WARNING, logger="ml.features"):
            ml_features._sanitize_close(s, "mymetric")
        warning_msgs = [
            r.message for r in caplog.records if "ml_feature_bad_context_row" in r.message
        ]
        assert len(warning_msgs) == 1
        # Message should mention the count of bad rows
        assert "3" in warning_msgs[0]
        assert "mymetric" in warning_msgs[0]


# ─────────────────────────────────────────────────────────────────────────────
# compute_vix_term_structure — bad input rows
# ─────────────────────────────────────────────────────────────────────────────

class TestVixTermStructureBadRows:
    def test_nan_vix9d_sets_missing(self, caplog):
        idx = _bdate(n=30)
        vix = _clean(idx, start=20.0)
        vix9d = _inject_bad(_clean(idx, start=18.0, seed=1), [5, 10], np.nan)
        vix3m = _clean(idx, start=22.0, seed=2)

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            slope, missing = ml_features.compute_vix_term_structure(
                vix, vix_short_close=vix9d, vix_long_close=vix3m
            )

        # Bad rows must be flagged as missing
        assert missing.iloc[5] == 1.0
        assert missing.iloc[10] == 1.0
        # Clean rows may be 0.0 (depending on staleness, not bad-mask)
        assert slope.between(-1.0, 1.0).all()
        assert any("ml_feature_bad_context_row" in r.message for r in caplog.records)

    def test_zero_vix3m_sets_missing(self, caplog):
        idx = _bdate(n=30)
        vix = _clean(idx, start=20.0)
        vix9d = _clean(idx, start=18.0, seed=1)
        vix3m = _inject_bad(_clean(idx, start=22.0, seed=2), [15], 0.0)

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            _, missing = ml_features.compute_vix_term_structure(
                vix, vix_short_close=vix9d, vix_long_close=vix3m
            )

        assert missing.iloc[15] == 1.0
        assert any("ml_feature_bad_context_row" in r.message for r in caplog.records)

    def test_slope_in_range_despite_bad_inputs(self):
        """slope must stay within [-1, 1] even when source rows are corrupt."""
        idx = _bdate(n=30)
        vix = _clean(idx, start=20.0)
        vix9d = _inject_bad(_clean(idx, start=18.0, seed=1), [0, 5, 29], 0.0)
        vix3m = _inject_bad(_clean(idx, start=22.0, seed=2), [3, 20], np.nan)

        slope, _ = ml_features.compute_vix_term_structure(
            vix, vix_short_close=vix9d, vix_long_close=vix3m
        )
        assert slope.between(-1.0, 1.0).all(), f"slope out of range: {slope.describe()}"
        assert slope.notna().all()

    def test_clean_positions_not_flagged_as_missing(self):
        """Positions with good data must NOT have missing=1.0 (unless stale)."""
        idx = _bdate(n=30)
        vix = _clean(idx, start=20.0)
        # Only position 0 is bad
        vix9d = _inject_bad(_clean(idx, start=18.0, seed=1), [0], np.nan)
        vix3m = _clean(idx, start=22.0, seed=2)

        _, missing = ml_features.compute_vix_term_structure(
            vix, vix_short_close=vix9d, vix_long_close=vix3m
        )
        # All rows from index 1 onwards should be clean (missing=0 if not stale)
        assert (missing.iloc[1:] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# compute_credit_stress — bad input rows (HYG / LQD)
# ─────────────────────────────────────────────────────────────────────────────

class TestCreditStressBadRows:
    def test_nan_hyg_sets_missing(self, caplog):
        idx = _bdate(n=40)
        hy = _inject_bad(_clean(idx, start=80.0, seed=0), [8, 20], np.nan)
        ig = _clean(idx, start=110.0, seed=1)

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            score, missing = ml_features.compute_credit_stress(
                idx, hy_close=hy, ig_close=ig
            )

        assert missing.iloc[8] == 1.0
        assert missing.iloc[20] == 1.0
        assert score.between(0.0, 1.0).all()
        assert any("ml_feature_bad_context_row" in r.message for r in caplog.records)

    def test_zero_lqd_sets_missing(self, caplog):
        idx = _bdate(n=40)
        hy = _clean(idx, start=80.0, seed=0)
        ig = _inject_bad(_clean(idx, start=110.0, seed=1), [12], 0.0)

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            score, missing = ml_features.compute_credit_stress(
                idx, hy_close=hy, ig_close=ig
            )

        assert missing.iloc[12] == 1.0
        assert score.between(0.0, 1.0).all()

    def test_score_in_range_with_bad_hyg_only(self):
        """HYG-only path with a bad row must still produce valid scores."""
        idx = _bdate(n=40)
        hy = _inject_bad(_clean(idx, start=80.0, seed=0), [0, 39], 0.0)

        score, _ = ml_features.compute_credit_stress(idx, hy_close=hy)
        assert score.between(0.0, 1.0).all()
        assert score.notna().all()

    def test_bad_hyg_and_ig_both_warn(self, caplog):
        idx = _bdate(n=30)
        hy = _inject_bad(_clean(idx, start=80.0, seed=0), [5], np.nan)
        ig = _inject_bad(_clean(idx, start=110.0, seed=1), [5], 0.0)

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            ml_features.compute_credit_stress(idx, hy_close=hy, ig_close=ig)

        warn_msgs = [
            r.message for r in caplog.records if "ml_feature_bad_context_row" in r.message
        ]
        # Expect one warning per bad series (hy_close + ig_close)
        assert len(warn_msgs) == 2

    def test_clean_positions_not_flagged(self):
        idx = _bdate(n=40)
        hy = _inject_bad(_clean(idx, start=80.0, seed=0), [0], np.nan)
        ig = _clean(idx, start=110.0, seed=1)

        _, missing = ml_features.compute_credit_stress(idx, hy_close=hy, ig_close=ig)
        # All positions after index 0 should be clean (missing=0)
        assert (missing.iloc[1:] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# compute_market_breadth — bad input rows (NYAD)
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketBreadthBadRows:
    def test_nan_nyad_sets_missing(self, caplog):
        idx = _bdate(n=40)
        breadth = _inject_bad(
            pd.Series(5000.0 + np.arange(40, dtype=float), index=idx),
            [10, 25],
            np.nan,
        )

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            score, missing = ml_features.compute_market_breadth(
                idx, breadth_close=breadth
            )

        assert missing.iloc[10] == 1.0
        assert missing.iloc[25] == 1.0
        assert score.between(0.0, 1.0).all()
        assert any("ml_feature_bad_context_row" in r.message for r in caplog.records)

    def test_zero_nyad_sets_missing(self, caplog):
        idx = _bdate(n=40)
        breadth = _inject_bad(
            pd.Series(5000.0 + np.arange(40, dtype=float), index=idx),
            [7],
            0.0,
        )

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            score, missing = ml_features.compute_market_breadth(
                idx, breadth_close=breadth
            )

        assert missing.iloc[7] == 1.0
        assert score.between(0.0, 1.0).all()

    def test_score_in_range_despite_corrupt_nyad(self):
        idx = _bdate(n=40)
        breadth = _inject_bad(
            _clean(idx, start=5000.0, vol=50.0),
            [0, 10, 20, 39],
            0.0,
        )
        score, _ = ml_features.compute_market_breadth(idx, breadth_close=breadth)
        assert score.between(0.0, 1.0).all()
        assert score.notna().all()

    def test_clean_positions_not_flagged(self):
        idx = _bdate(n=40)
        breadth = _inject_bad(
            pd.Series(5000.0 + np.arange(40, dtype=float), index=idx),
            [39],
            np.nan,
        )
        _, missing = ml_features.compute_market_breadth(idx, breadth_close=breadth)
        # Only last position is bad; all others must be clean
        assert (missing.iloc[:-1] == 0.0).all()
        assert missing.iloc[39] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_rates_level — bad input rows (TNX)
# ─────────────────────────────────────────────────────────────────────────────

class TestRatesLevelBadRows:
    def test_nan_tnx_sets_missing(self, caplog):
        idx = _bdate(n=30)
        rates = _inject_bad(pd.Series(45.0, index=idx), [4, 18], np.nan)

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            norm, missing = ml_features.compute_rates_level(idx, rates_close=rates)

        assert missing.iloc[4] == 1.0
        assert missing.iloc[18] == 1.0
        assert norm.between(0.0, 1.0).all()
        assert any("ml_feature_bad_context_row" in r.message for r in caplog.records)

    def test_zero_tnx_sets_missing(self, caplog):
        idx = _bdate(n=30)
        rates = _inject_bad(pd.Series(45.0, index=idx), [0], 0.0)

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            norm, missing = ml_features.compute_rates_level(idx, rates_close=rates)

        assert missing.iloc[0] == 1.0
        assert norm.between(0.0, 1.0).all()

    def test_norm_in_range_despite_corrupt_tnx(self):
        idx = _bdate(n=30)
        rates = _inject_bad(pd.Series(45.0, index=idx), [0, 15, 29], np.nan)
        norm, _ = ml_features.compute_rates_level(idx, rates_close=rates)
        assert norm.between(0.0, 1.0).all()
        assert norm.notna().all()

    def test_warning_includes_label(self, caplog):
        idx = _bdate(n=10)
        rates = _inject_bad(pd.Series(45.0, index=idx), [3], 0.0)

        with caplog.at_level(logging.WARNING, logger="ml.features"):
            ml_features.compute_rates_level(idx, rates_close=rates)

        warns = [
            r.message for r in caplog.records if "ml_feature_bad_context_row" in r.message
        ]
        assert len(warns) == 1
        assert "rates_close" in warns[0]

    def test_clean_positions_not_flagged(self):
        idx = _bdate(n=30)
        rates = _inject_bad(pd.Series(45.0, index=idx), [29], np.nan)
        _, missing = ml_features.compute_rates_level(idx, rates_close=rates)
        assert (missing.iloc[:-1] == 0.0).all()
        assert missing.iloc[29] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-cutting: bad rows never reach the model as raw values
# ─────────────────────────────────────────────────────────────────────────────

class TestBadRowsNeverPassRaw:
    """The golden-rule invariant: a zero or NaN close must NEVER produce
    a raw NaN or out-of-range value in the feature output."""

    def test_vix_term_no_nan_in_slope(self):
        idx = _bdate(n=30)
        vix = _clean(idx, start=20.0)
        vix9d = _inject_bad(_clean(idx, start=18.0), list(range(30)), 0.0)
        vix3m = _inject_bad(_clean(idx, start=22.0), list(range(30)), np.nan)

        slope, missing = ml_features.compute_vix_term_structure(
            vix, vix_short_close=vix9d, vix_long_close=vix3m
        )
        assert slope.notna().all(), "NaN leaked into slope"
        assert slope.between(-1.0, 1.0).all()
        assert (missing == 1.0).all()

    def test_credit_stress_no_nan_in_score(self):
        idx = _bdate(n=30)
        hy = _inject_bad(_clean(idx, start=80.0), list(range(30)), np.nan)

        score, missing = ml_features.compute_credit_stress(idx, hy_close=hy)
        assert score.notna().all(), "NaN leaked into credit stress score"
        assert score.between(0.0, 1.0).all()
        assert (missing == 1.0).all()

    def test_breadth_no_nan_in_score(self):
        idx = _bdate(n=30)
        breadth = _inject_bad(_clean(idx, start=5000.0), list(range(30)), 0.0)

        score, missing = ml_features.compute_market_breadth(idx, breadth_close=breadth)
        assert score.notna().all(), "NaN leaked into breadth score"
        assert score.between(0.0, 1.0).all()
        assert (missing == 1.0).all()

    def test_rates_no_nan_in_norm(self):
        idx = _bdate(n=30)
        rates = _inject_bad(pd.Series(45.0, index=idx), list(range(30)), 0.0)

        norm, missing = ml_features.compute_rates_level(idx, rates_close=rates)
        assert norm.notna().all(), "NaN leaked into rates norm"
        assert norm.between(0.0, 1.0).all()
        assert (missing == 1.0).all()
