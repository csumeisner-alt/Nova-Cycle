"""
NovaCycle Short-Trend ML Model
================================
3-layer MLP (scikit-learn MLPClassifier) for short-term (5-min) VOO signals.
Replaces the previous TensorFlow/Keras implementation with a lightweight
scikit-learn equivalent — same feature set, same time-decay sample weights.

NOTE: "Model currently trained only for ticker='VOO'. Multi-ticker support
      will be added later."

Features (from 5-min candles, including extended hours):
  - RSI, StochRSI_K, StochRSI_D, Stochastic_K, Stochastic_D
  - Bollinger %B, Bollinger bandwidth
  - ATR, ATR ratio (ATR / close)
  - Recent returns: 1h (12 bars), 2h (24 bars), 4h (48 bars), 1d (78 bars)
  - is_extended_hours (0/1)
  - GapPercent, LiquidityScore
  - Return_overnight = (open - prev_close) / prev_close
  - Volume ratio vs 20-bar rolling avg

Architecture (equivalent to 128→64→32→1 Keras dense net):
  MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                solver='adam', max_iter=200)

Target:
  y = 1 if return_1h > 0.003 else 0   [0.3% move in next hour]

Time-decay:
  Weight = (0.5 if extended else 1.0) × exp(-LAMBDA_SHORT × age_in_minutes)
"""

import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import settings
from ml import features as ml_features
from ml.model_health import check_model_degeneracy

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "short_trend_model.pkl"

FEATURE_NAMES = [
    "rsi",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "stoch_k",
    "stoch_d",
    "bb_pct_b",
    "bb_bandwidth",
    "atr",
    "atr_ratio",
    "return_1h",
    "return_2h",
    "return_4h",
    "return_1d",
    "is_extended",
    "gap_percent",
    "liquidity_score",
    "return_overnight",
    "volume_ratio",
    # Additive VOO-specific features (in-memory only)
    "volatility_regime_enc",
    "macro_sensitivity_score",
    "macro_override_flag",
    "gap_momentum",
    "gap_momentum_class_enc",
    "liquidity_compression_score",
]

N_FEATURES = len(FEATURE_NAMES)


class ShortTrendModel:
    """scikit-learn MLP predicting 1-hour forward return > 0.3%."""

    def __init__(self):
        self.model: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self._model_loaded = False
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────────────

    def build_features(
        self,
        df: pd.DataFrame,
        indicators: dict,
        liquidity_score: float = 1.0,
        gap_percent: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix from 5-min candle DataFrame.

        Extended-hours weight:
          base_weight = 0.5 if is_extended_hours else 1.0

        Time-decay:
          Weight = base_weight × exp(-LAMBDA_SHORT × age_in_minutes)

        Returns:
            (X: np.ndarray [n, N_FEATURES], weights: np.ndarray [n])
        """
        rows = []
        weights = []

        now = pd.Timestamp.utcnow().tz_localize(None)

        close = df["close"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)
        open_col = df["open"] if "open" in df.columns else close

        rsi_s = indicators.get("rsi", pd.Series(50.0, index=df.index))
        stoch = indicators.get("stoch", {
            "k": pd.Series(50.0, index=df.index),
            "d": pd.Series(50.0, index=df.index),
        })
        stoch_rsi = indicators.get("stoch_rsi", {
            "k": pd.Series(50.0, index=df.index),
            "d": pd.Series(50.0, index=df.index),
        })
        bb = indicators.get("bollinger", {
            "pct_b": pd.Series(0.5, index=df.index),
            "bandwidth": pd.Series(0.0, index=df.index),
        })
        atr_s = indicators.get("atr_all", pd.Series(0.0, index=df.index))

        vol_avg20 = volume.rolling(20).mean()

        # ── Additive features (vectorized, in-memory) ─────────────────────────
        liq_class = df["liquidity_class"] if "liquidity_class" in df.columns else None
        vol_regimes = ml_features.compute_volatility_regime(
            close, atr=atr_s, liquidity_class=liq_class
        )
        vol_regime_enc = ml_features.encode_volatility_regime(vol_regimes)
        macro_sens = ml_features.compute_macro_sensitivity(
            close,
            open_=open_col,
            vix_regime=indicators.get("vix_regime"),
            spx_futures_close=indicators.get("spx_futures_close"),
        )
        macro_flag = ml_features.macro_override_flag(
            df.index,
            close=close,
            open_=open_col,
            vix_regime=indicators.get("vix_regime"),
            volatility_regime=vol_regimes,
        )
        gap_momentum_s, gap_momentum_cls = ml_features.compute_gap_momentum_features(df)
        liq_compression = ml_features.compute_liquidity_compression_score(df)

        # Bars per period at 5-min resolution
        BARS_1H = 12
        BARS_2H = 24
        BARS_4H = 48
        BARS_1D = 78  # ~6.5h trading day

        for i, (ts, row) in enumerate(df.iterrows()):
            try:
                c = float(row["close"])
                if c <= 0:
                    continue

                ts_naive = ts
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    ts_naive = ts.replace(tzinfo=None)

                is_ext = int(bool(row.get("is_extended_hours", False)))

                def _get(series, idx):
                    try:
                        return float(series.iloc[idx])
                    except Exception:
                        return 50.0

                rsi_val  = _get(rsi_s, i)
                sk_val   = _get(stoch["k"], i)
                sd_val   = _get(stoch["d"], i)
                srk_val  = _get(stoch_rsi["k"], i)
                srd_val  = _get(stoch_rsi["d"], i)
                bb_pb    = _get(bb["pct_b"], i)
                bb_bw    = _get(bb["bandwidth"], i)
                atr_val  = _get(atr_s, i)
                atr_ratio = (atr_val / c) if c != 0 else 0.0

                def _ret(n_bars):
                    if i >= n_bars:
                        prev = float(close.iloc[i - n_bars])
                        return (c - prev) / prev if prev != 0 else 0.0
                    return 0.0

                ret_1h = _ret(BARS_1H)
                ret_2h = _ret(BARS_2H)
                ret_4h = _ret(BARS_4H)
                ret_1d = _ret(BARS_1D)

                if i >= 1:
                    prev_c = float(close.iloc[i - 1])
                    o_val = float(open_col.iloc[i])
                    ret_overnight = (o_val - prev_c) / prev_c if prev_c != 0 else 0.0
                else:
                    ret_overnight = 0.0

                vol_a = float(vol_avg20.iloc[i]) if not np.isnan(vol_avg20.iloc[i]) else 0.0
                vol_r = float(volume.iloc[i])
                vol_ratio = (vol_r / vol_a) if vol_a > 0 else 1.0

                feature_row = [
                    rsi_val / 100.0,
                    srk_val / 100.0,
                    srd_val / 100.0,
                    sk_val / 100.0,
                    sd_val / 100.0,
                    bb_pb,
                    bb_bw,
                    atr_val,
                    atr_ratio,
                    ret_1h,
                    ret_2h,
                    ret_4h,
                    ret_1d,
                    float(is_ext),
                    gap_percent / 100.0,
                    liquidity_score,
                    ret_overnight,
                    vol_ratio,
                    float(vol_regime_enc.iloc[i]),
                    float(macro_sens.iloc[i]),
                    float(macro_flag.iloc[i]),
                    float(gap_momentum_s.iloc[i]) / 100.0,
                    float(gap_momentum_cls.iloc[i]),
                    float(liq_compression.iloc[i]),
                ]
                rows.append(feature_row)

                base_w = 0.5 if is_ext else 1.0
                age_min = max(0.0, (now - ts_naive).total_seconds() / 60.0)
                w = base_w * math.exp(-settings.LAMBDA_SHORT * age_min)
                weights.append(max(w, 1e-6))

            except Exception as exc:
                logger.debug("Skipping 5-min row %s: %s", ts, exc)
                continue

        if not rows:
            return np.array([]).reshape(0, N_FEATURES), np.array([])

        return np.array(rows, dtype=np.float32), np.array(weights, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, indicators: dict) -> dict:
        """
        Train the MLPClassifier.

        Target:
          y = 1 if (Close[t+12] - Close[t]) / Close[t] > 0.003 else 0

        Saves scaler + model to ml/models/short_trend_model.pkl.

        Returns:
            {"accuracy": float, "val_accuracy": float}
        """
        try:
            BARS_1H = 12
            df = df.copy()
            df["future_close"] = df["close"].shift(-BARS_1H)
            df.dropna(subset=["future_close"], inplace=True)
            df["label"] = ((df["future_close"] - df["close"]) / df["close"] > 0.003).astype(int)

            X, sample_weights = self.build_features(df, indicators)
            y = df["label"].values[: len(X)]

            if len(X) < 100:
                logger.warning(
                    "Not enough 5-min data to train short-trend model (%d rows)", len(X)
                )
                return {"accuracy": 0.0, "val_accuracy": 0.0}

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split 85/15 for validation
            split = int(len(X_scaled) * 0.85)
            X_train, X_val = X_scaled[:split], X_scaled[split:]
            y_train, y_val = y[:split], y[split:]
            w_train = sample_weights[:split]

            model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                verbose=False,
            )
            # MLPClassifier does not support sample_weight — approximate the
            # recency weighting by oversampling the most recent (highest-
            # weight) rows instead of passing weights to fit().
            try:
                threshold = np.percentile(w_train, 75) if len(w_train) else 0.0
                recent_mask = w_train >= threshold
                if 0 < recent_mask.sum() < len(X_train):
                    X_fit = np.vstack([X_train, X_train[recent_mask]])
                    y_fit = np.concatenate([y_train, y_train[recent_mask]])
                else:
                    X_fit, y_fit = X_train, y_train
            except Exception:
                X_fit, y_fit = X_train, y_train
            model.fit(X_fit, y_fit)

            train_acc = float(model.score(X_train, y_train))
            val_acc = float(model.score(X_val, y_val)) if len(X_val) > 0 else 0.0

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({"model": model, "scaler": scaler}, f)

            self.model = model
            self.scaler = scaler
            self._model_loaded = True

            # Degeneracy health check: verify predictions actually vary across
            # the training set (a constant predictor can still report the
            # base-rate accuracy while being useless).
            degenerate, degeneracy_reason = check_model_degeneracy(model, X_scaled)
            if degenerate:
                logger.error(
                    "ml_model_degenerate model=short_trend reason=%s",
                    degeneracy_reason,
                )

            logger.info(
                "Short-trend MLP trained: acc=%.4f  val_acc=%.4f", train_acc, val_acc
            )
            return {
                "accuracy": train_acc,
                "val_accuracy": val_acc,
                "degenerate": degenerate,
                "degeneracy_reason": degeneracy_reason,
            }

        except Exception as exc:
            logger.error("Short-trend model training error: %s", exc)
            return {"accuracy": 0.0, "val_accuracy": 0.0}

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> float:
        """
        Return BUY probability in [0, 1].

        Loads model from disk if not already loaded.
        """
        try:
            if not self._model_loaded:
                self.load_model()

            if self.model is None:
                logger.warning("Short-trend model not available; returning 0.5")
                return 0.5

            if features.ndim == 1:
                features = features.reshape(1, -1)

            if self.scaler is not None:
                features = self.scaler.transform(features)

            probs = self.model.predict_proba(features)
            # predict_proba returns [[p_class0, p_class1]]
            return float(probs[0][1]) if probs.shape[1] > 1 else float(probs[0][0])

        except Exception as exc:
            logger.error("Short-trend predict error: %s", exc)
            return 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # Model persistence
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        """Load MLP + scaler from ml/models/short_trend_model.pkl."""
        try:
            if MODEL_PATH.exists():
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                model = data["model"]
                scaler = data.get("scaler")
                # Guard: models pickled before the feature-set extension have
                # a smaller feature count. Never crash prediction on them —
                # discard, log, and fall back to neutral until retraining.
                n_in = getattr(scaler, "n_features_in_", None) or getattr(
                    model, "n_features_in_", None
                )
                if n_in is not None and int(n_in) != N_FEATURES:
                    logger.warning(
                        "ml_model_stale model=short_trend expected_features=%d "
                        "found=%d action=discard_await_retrain",
                        N_FEATURES, int(n_in),
                    )
                    self.model = None
                    self.scaler = None
                    self._model_loaded = True
                    return False
                self.model = model
                self.scaler = scaler
                self._model_loaded = True
                logger.info("Short-trend model loaded from %s", MODEL_PATH)
                return True
            else:
                logger.info("No short-trend model file at %s — will train on first run", MODEL_PATH)
                self._model_loaded = True
                return False
        except Exception as exc:
            logger.error("Error loading short-trend model: %s", exc)
            self._model_loaded = True
            return False

    def is_neutral_fallback(self) -> bool:
        """
        True when the model is unavailable (missing, stale, or failed to load)
        and predict() would return the neutral 0.5 fallback.
        """
        if not self._model_loaded:
            self.load_model()
        return self.model is None

    def build_latest_features(
        self,
        df: pd.DataFrame,
        indicators: dict,
        liquidity_score: float = 1.0,
        gap_percent: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Build the feature vector for the most recent 5-min bar."""
        try:
            X, _ = self.build_features(df, indicators, liquidity_score, gap_percent)
            if len(X) == 0:
                return None
            return X[-1:].astype(np.float32)
        except Exception as exc:
            logger.error("build_latest_features (short) error: %s", exc)
            return None
