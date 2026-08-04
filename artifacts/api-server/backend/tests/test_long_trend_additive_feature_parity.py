"""
Regression tests: long-trend additive-feature train/inference parity.

Root cause (same bug class as the _return_* alignment fix): build_features()
recomputed the four additive rolling-window features —
volatility_regime_enc, macro_sensitivity_score, macro_override_flag and
overnight_return_weighted — on whatever df was passed.  train() passes the
meaningful-move-filtered subset (non-contiguous dates), so rolling windows
spanned filtered rows rather than real trading days, while inference passes
the full df.  The resulting train/inference feature mismatch inverted model
behavior in calm regimes and produced strongly negative OOS lift.

Fix: train() pre-computes _vol_regime_enc, _macro_sens, _macro_flag and
_overnight_w on the full unfiltered df before the filter is applied;
build_features() consumes these columns when present and falls back to
recomputing only when absent (inference path, full df).

Covers:
  - Full feature-vector parity: training-path rows == inference-path rows for
    the same timestamps (all 19 features, not just returns).
  - build_features() prefers the pre-computed columns when present.
"""

import numpy as np
import pandas as pd

from ml.long_trend import LongTrendModel, FEATURE_NAMES


def _make_daily_df(n: int = 400, seed: int = 7, drift: float = 0.03, vol: float = 2.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    price = np.maximum(300.0 + np.cumsum(rng.normal(drift, vol, n)), 1.0)
    return pd.DataFrame({
        "open":  price - rng.uniform(0, 0.5, n),
        "high":  price + rng.uniform(0, 1.0, n),
        "low":   price - rng.uniform(0, 1.5, n),
        "close": price,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "is_extended_hours": False,
    }, index=idx)


def _make_indicators(df: pd.DataFrame) -> dict:
    close = df["close"]
    return {
        "sma20": close.rolling(20).mean(),
        "sma50": close.rolling(50).mean(),
        "sma200": close.rolling(200).mean(),
        "macd": close.ewm(span=12).mean() - close.ewm(span=26).mean(),
        "macd_signal": (close.ewm(span=12).mean() - close.ewm(span=26).mean()).ewm(span=9).mean(),
        "adx": pd.Series(20.0, index=df.index),
        "atr": (df["high"] - df["low"]).rolling(14).mean(),
        "vix_regime": pd.Series("NORMAL", index=df.index),
        "vix_level": pd.Series(18.0, index=df.index),
        "vix_change_5d": pd.Series(0.0, index=df.index),
        "vix_percentile_1y": pd.Series(0.5, index=df.index),
        "vix_missing": pd.Series(False, index=df.index),
    }


def _precompute_train_columns(df: pd.DataFrame, indicators: dict) -> pd.DataFrame:
    """Replicate train()'s full-df precompute step."""
    from ml import features as ml_features

    df = df.copy()
    df["_return_5d"] = df["close"].pct_change(5)
    df["_return_10d"] = df["close"].pct_change(10)
    df["_return_20d"] = df["close"].pct_change(20)
    df["_vol_avg20"] = df["volume"].rolling(20).mean()
    close, open_ = df["close"], df["open"]
    vol_regimes = ml_features.compute_volatility_regime(
        close, atr=indicators["atr"], liquidity_class=None
    )
    df["_vol_regime_enc"] = ml_features.encode_volatility_regime(vol_regimes)
    df["_macro_sens"] = ml_features.compute_macro_sensitivity(
        close, open_=open_, vix_regime=indicators["vix_regime"],
        spx_futures_close=None,
    )
    df["_macro_flag"] = ml_features.macro_override_flag(
        df.index, close=close, open_=open_,
        vix_regime=indicators["vix_regime"], volatility_regime=vol_regimes,
    )
    df["_overnight_w"] = ml_features.compute_overnight_return_weighted(open_, close)
    return df


def test_train_path_features_match_inference_path():
    """For every timestamp that survives the meaningful-move filter, the
    training-path feature row (filtered subset + precomputed columns) must
    equal the inference-path feature row (full df, no precomputed columns)."""
    df = _make_daily_df()
    indicators = _make_indicators(df)
    model = LongTrendModel()

    # Inference path: full df, no helper columns.
    X_inf, _, pos_inf = model.build_features(df, indicators)
    inf_by_ts = {df.index[p]: X_inf[j] for j, p in enumerate(pos_inf)}

    # Training path: precompute on full df, then apply the meaningful-move filter.
    pre = _precompute_train_columns(df, indicators)
    pre["future_close"] = pre["close"].shift(-21)
    pre = pre.dropna(subset=["future_close"])
    pre["fwd"] = pre["future_close"] / pre["close"] - 1.0
    filt = pre[(pre["fwd"] >= 0.02) | (pre["fwd"] <= -0.02)].copy()
    assert len(filt) < len(df), "filter must remove rows for the test to be meaningful"
    trimmed = {k: v.reindex(filt.index) if isinstance(v, pd.Series) else v
               for k, v in indicators.items()}
    X_tr, _, pos_tr = model.build_features(filt, trimmed)

    mismatches = {}
    for j, p in enumerate(pos_tr):
        ts = filt.index[p]
        assert ts in inf_by_ts
        diff = np.abs(X_tr[j] - inf_by_ts[ts])
        for k, d in enumerate(diff):
            if d > 1e-6:
                mismatches.setdefault(FEATURE_NAMES[k], 0)
                mismatches[FEATURE_NAMES[k]] += 1
    assert not mismatches, f"train/inference feature mismatch: {mismatches}"


def test_build_features_prefers_precomputed_columns():
    """When the _vol_regime_enc/_macro_sens/_macro_flag/_overnight_w columns
    are present, build_features must use them verbatim."""
    df = _make_daily_df(n=120, seed=3)
    indicators = _make_indicators(df)
    model = LongTrendModel()

    df2 = df.copy()
    df2["_vol_regime_enc"] = 2.0
    df2["_macro_sens"] = 0.123
    df2["_macro_flag"] = 1.0
    df2["_overnight_w"] = -0.456

    X, _, _ = model.build_features(df2, indicators)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert np.allclose(X[:, idx["volatility_regime_enc"]], 2.0)
    assert np.allclose(X[:, idx["macro_sensitivity_score"]], 0.123)
    assert np.allclose(X[:, idx["macro_override_flag"]], 1.0)
    assert np.allclose(X[:, idx["overnight_return_weighted"]], -0.456, atol=1e-6)


def test_partial_precompute_falls_back_safely():
    """A caller providing only some precomputed columns (e.g. _vol_regime_enc
    but not _macro_flag) must still get a valid macro_flag via the fallback
    recomputation path — no crash, values in {0, 1}."""
    df = _make_daily_df(n=120, seed=5)
    indicators = _make_indicators(df)
    model = LongTrendModel()

    df2 = df.copy()
    df2["_vol_regime_enc"] = 1.0  # present
    # _macro_sens, _macro_flag, _overnight_w deliberately absent

    X, _, _ = model.build_features(df2, indicators)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert len(X) > 0
    assert np.allclose(X[:, idx["volatility_regime_enc"]], 1.0)
    flags = X[:, idx["macro_override_flag"]]
    assert set(np.unique(flags)).issubset({0.0, 1.0})
