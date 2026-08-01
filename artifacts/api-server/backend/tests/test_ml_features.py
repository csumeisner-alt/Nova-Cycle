"""Tests for the VOO ML feature-engineering upgrade."""

import numpy as np
import pandas as pd
import pytest

from ml import features as ml_features
from ml.long_trend import LongTrendModel, FEATURE_NAMES as LONG_FEATURES
from ml.short_trend import ShortTrendModel, FEATURE_NAMES as SHORT_FEATURES, N_FEATURES


def _daily_df(n=120):
    idx = pd.bdate_range("2026-01-02", periods=n)
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(100.0),
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
        "is_extended_hours": False,
        "session_type": "regular",
    }, index=idx)


def _fivemin_df(n=300):
    idx = pd.date_range("2026-07-20 13:30", periods=n, freq="5min")
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.05, n)), index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(100.0),
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": rng.uniform(1e4, 5e4, n),
        "is_extended_hours": False,
        "session_type": "regular",
        "gap_percent": 0.5,
        "gap_type": "none",
    }, index=idx)


class TestVolatilityRegime:
    def test_labels_valid(self):
        df = _daily_df()
        regimes = ml_features.compute_volatility_regime(df["close"])
        assert set(regimes.unique()) <= {"calm", "trending", "macro_shock", "compressed"}
        assert len(regimes) == len(df)

    def test_thin_liquidity_forces_compressed(self):
        df = _daily_df(60)
        liq = pd.Series("thin", index=df.index)
        regimes = ml_features.compute_volatility_regime(df["close"], liquidity_class=liq)
        # thin liquidity → compressed unless macro_shock wins
        assert (regimes.isin(["compressed", "macro_shock"])).all()

    def test_encoding_default_on_unknown(self):
        s = pd.Series(["calm", "bogus", "macro_shock"])
        enc = ml_features.encode_volatility_regime(s)
        assert enc.tolist() == [0.0, 0.0, 3.0]


class TestMacroSensitivity:
    def test_range_and_fallback(self):
        df = _daily_df()
        score = ml_features.compute_macro_sensitivity(df["close"], open_=df["open"])
        assert ((score >= 0.0) & (score <= 1.0)).all()

    def test_vix_regime_raises_score(self):
        df = _daily_df()
        low = ml_features.compute_macro_sensitivity(
            df["close"], open_=df["open"],
            vix_regime=pd.Series("LOW", index=df.index))
        extreme = ml_features.compute_macro_sensitivity(
            df["close"], open_=df["open"],
            vix_regime=pd.Series("EXTREME", index=df.index))
        assert extreme.mean() > low.mean()

    def test_failure_returns_default(self):
        score = ml_features.compute_macro_sensitivity(pd.Series(dtype=float))
        assert len(score) == 0  # empty in, empty out, no crash


class TestMacroOverrideFlag:
    def test_default_zero_without_inputs(self):
        df = _daily_df(30)
        flag = ml_features.macro_override_flag(df.index)
        assert (flag == 0.0).all()

    def test_fires_on_extreme_vix_plus_big_overnight_move(self):
        df = _daily_df(30)
        open_ = df["close"].shift(1).fillna(100.0) * 1.05  # +5% overnight gaps
        vix = pd.Series("EXTREME", index=df.index)
        flag = ml_features.macro_override_flag(
            df.index, close=df["close"], open_=open_, vix_regime=vix
        )
        assert flag.iloc[1:].eq(1.0).all()  # first bar has no prev close

    def test_not_fired_when_vix_extreme_but_calm_overnight(self):
        df = _daily_df(30)
        open_ = df["close"].shift(1).fillna(100.0)  # zero overnight move
        vix = pd.Series("EXTREME", index=df.index)
        flag = ml_features.macro_override_flag(
            df.index, close=df["close"], open_=open_, vix_regime=vix
        )
        assert (flag == 0.0).all()

    def test_not_fired_on_big_move_with_low_vix(self):
        df = _daily_df(30)
        open_ = df["close"].shift(1).fillna(100.0) * 1.05
        vix = pd.Series("LOW", index=df.index)
        flag = ml_features.macro_override_flag(
            df.index, close=df["close"], open_=open_, vix_regime=vix
        )
        assert (flag == 0.0).all()

    def test_fires_on_macro_shock_regime(self):
        df = _daily_df(30)
        regimes = pd.Series("calm", index=df.index)
        regimes.iloc[10:15] = "macro_shock"
        flag = ml_features.macro_override_flag(
            df.index, close=df["close"], volatility_regime=regimes
        )
        assert flag.iloc[10:15].eq(1.0).all()
        assert flag.drop(flag.index[10:15]).eq(0.0).all()

    def test_binary_values_only(self):
        df = _daily_df(60)
        vix = pd.Series("HIGH", index=df.index)
        regimes = ml_features.compute_volatility_regime(df["close"])
        flag = ml_features.macro_override_flag(
            df.index, close=df["close"], open_=df["open"],
            vix_regime=vix, volatility_regime=regimes,
        )
        assert set(flag.unique()) <= {0.0, 1.0}


class TestGapMomentum:
    def test_direction_sign(self):
        df = _fivemin_df(50)
        # Force first candle of the day bullish
        df.iloc[0, df.columns.get_loc("open")] = 99.0
        df.iloc[0, df.columns.get_loc("close")] = 101.0
        gm, cls = ml_features.compute_gap_momentum_features(df)
        assert (gm == 0.5).all()  # gap 0.5 × direction +1
        assert set(cls.unique()) <= {0.0, 1.0, 2.0}

    def test_no_gap_column(self):
        df = _fivemin_df(20).drop(columns=["gap_percent"])
        gm, cls = ml_features.compute_gap_momentum_features(df)
        assert (gm == 0.0).all() and (cls == 0.0).all()

    def test_classify_scalar(self):
        assert ml_features.classify_gap_momentum(0.05) == "weak"
        assert ml_features.classify_gap_momentum(0.5) == "medium"
        assert ml_features.classify_gap_momentum(1.5) == "strong"


class TestLiquidityCompression:
    def test_volume_deviation(self):
        df = _fivemin_df(60)
        score = ml_features.compute_liquidity_compression_score(df)
        assert ((score >= 0.0) & (score <= 1.0)).all()

    def test_reuses_ingestion_column(self):
        df = _fivemin_df(20)
        df["liquidity_compression"] = 1.0  # fully healthy per ingestion
        score = ml_features.compute_liquidity_compression_score(df)
        assert (score == 0.0).all()  # inverted: healthy → no compression


class TestOvernightWeighted:
    def test_zero_when_flat(self):
        idx = pd.bdate_range("2026-01-02", periods=30)
        close = pd.Series(100.0, index=idx)
        open_ = pd.Series(100.0, index=idx)
        w = ml_features.compute_overnight_return_weighted(open_, close)
        assert (w == 0.0).all()


class TestModelPipelines:
    def test_long_feature_matrix_width(self):
        df = _daily_df()
        X, w, pos = LongTrendModel().build_features(df, {})
        assert X.shape[1] == len(LONG_FEATURES)
        assert len(w) == len(X) == len(pos)
        assert np.isfinite(X).all()

    def test_short_feature_matrix_width(self):
        df = _fivemin_df()
        X, w = ShortTrendModel().build_features(df, {})
        assert X.shape[1] == N_FEATURES == len(SHORT_FEATURES)
        assert len(w) == len(X)
        assert np.isfinite(X).all()

    def test_stale_model_discarded(self, tmp_path, monkeypatch):
        """A pickled model with the old feature count must be discarded, not crash."""
        import pickle
        from sklearn.preprocessing import StandardScaler
        from ml import short_trend as st

        old_n = N_FEATURES - 6
        scaler = StandardScaler().fit(np.zeros((5, old_n)))
        path = tmp_path / "short_trend_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": object(), "scaler": scaler}, f)
        monkeypatch.setattr(st, "MODEL_PATH", path)

        m = ShortTrendModel()
        assert m.load_model() is False
        assert m.model is None
        assert m.predict(np.zeros(N_FEATURES)) == 0.5
