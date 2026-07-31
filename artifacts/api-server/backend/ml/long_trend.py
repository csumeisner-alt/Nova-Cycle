"""
NovaCycle Long-Trend ML Model
==============================
XGBoost gradient-boosted classifier for long-term (daily) VOO trend.

NOTE: "Model currently trained only for ticker='VOO'. Multi-ticker support will be added later."

Features (all from regular-hours candles only):
  - SMA50/SMA200 ratio
  - MACD line, MACD signal line
  - ADX
  - VIX regime (label-encoded: LOW=0, NORMAL=1, HIGH=2, EXTREME=3)
  - Recent returns: 5d, 10d, 20d
  - Volume ratio (current / 20-day avg)
  - ATR (normalised by close)
  - SMA20 distance ((close - SMA20) / SMA20)

Target:
  label = 1 if forward_return_21d > 0 else 0

Time-decay sample weight:
  Weight(t) = exp(-LAMBDA_LONG × age_in_days)
"""

import logging
import math
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from ml import features as ml_features
from ml import calibration as ml_calibration
from ml.model_health import check_model_degeneracy

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "long_trend_model.pkl"

VIX_REGIME_MAP = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}

FEATURE_NAMES = [
    "sma50_200_ratio",
    "macd_line",
    "macd_signal",
    "adx",
    "vix_regime_enc",
    "return_5d",
    "return_10d",
    "return_20d",
    "volume_ratio",
    "atr_norm",
    "sma20_distance",
    # Additive VOO-specific features (in-memory only)
    "volatility_regime_enc",
    "macro_sensitivity_score",
    "macro_override_flag",
    "overnight_return_weighted",
]


class LongTrendModel:
    """XGBoost model predicting 21-day forward return direction for VOO."""

    def __init__(self):
        self.model = None
        self.calibrator = None
        self._model_loaded = False
        self._loaded_mtime: Optional[float] = None
        self._calibrator_mtime: Optional[float] = None
        self.calibration_base_rate: Optional[float] = None
        self._calibration_report_mtime: Optional[float] = None
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def _maybe_reload(self) -> None:
        """
        Load the model on first use, and reload it when the on-disk file has
        appeared or changed since the last load (e.g. after a retrain in
        another component of the same process).
        """
        try:
            mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
        except OSError:
            mtime = None

        if not self._model_loaded or mtime != self._loaded_mtime:
            self.load_model()
            self._loaded_mtime = mtime

        # Reload the calibrator when its file appears or changes (e.g. after
        # a retrain in the trainer component of the same process).
        try:
            cal_path = ml_calibration.CALIBRATOR_PATH
            cal_mtime = cal_path.stat().st_mtime if cal_path.exists() else None
        except OSError:
            cal_mtime = None
        if cal_mtime != self._calibrator_mtime:
            calibrator = ml_calibration.load_calibrator()
            self.calibrator = calibrator
            # Only pin the mtime when the load succeeded (or the file is
            # genuinely absent); a transient read failure must be retried on
            # the next prediction rather than silently disabling calibration.
            if calibrator is not None or cal_mtime is None:
                self._calibrator_mtime = cal_mtime

        # Reload the calibration report's positive rate when the report file
        # appears, changes, or disappears.  A deleted report (e.g. a failed
        # retrain that removed the old file) must reset the base rate so the
        # gauge falls back to the safe 0.5 neutral point rather than silently
        # retaining a stale rate.
        try:
            report_path = ml_calibration.calibration_report_path("long_trend")
            report_mtime = report_path.stat().st_mtime if report_path.exists() else None
        except OSError:
            report_mtime = None
        if report_mtime != self._calibration_report_mtime:
            report = ml_calibration.get_calibration_report("long_trend")
            rate = report.get("positive_rate") if isinstance(report, dict) else None
            try:
                rate = float(rate)
                self.calibration_base_rate = (
                    min(0.99, max(0.01, rate)) if 0.0 < rate < 1.0 else None
                )
            except (TypeError, ValueError):
                self.calibration_base_rate = None
            self._calibration_report_mtime = report_mtime

    def get_neutral_probability(self) -> float:
        """Return the calibrated probability that represents a normal outcome.

        A missing or invalid calibration report deliberately returns 0.5 so
        legacy models and neutral fallbacks retain the old behavior.
        """
        self._maybe_reload()
        return self.calibration_base_rate or 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────────────

    def build_features(
        self,
        df: pd.DataFrame,
        indicators: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the feature matrix from regular-hours daily data.

        Time-decay:
          Weight(t) = exp(-LAMBDA_LONG × age_in_days)
          where age_in_days = (today - timestamp).days

        Args:
            df:         VOO daily DataFrame (regular-hours only; indexed by timestamp)
            indicators: Output of TechnicalIndicators.compute_all()

        Returns:
            (X: np.ndarray [n, features], weights: np.ndarray [n])
        """
        rows = []
        weights = []

        now = pd.Timestamp.utcnow().tz_localize(None)

        # Ensure regular-hours only
        if "is_extended_hours" in df.columns:
            df = df[df["is_extended_hours"] == False].copy()

        close = df["close"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

        sma50 = indicators.get("sma50", pd.Series(dtype=float))
        sma200 = indicators.get("sma200", pd.Series(dtype=float))
        sma20 = indicators.get("sma20", pd.Series(dtype=float))
        macd = indicators.get("macd", pd.Series(dtype=float))
        macd_signal = indicators.get("macd_signal", pd.Series(dtype=float))
        adx = indicators.get("adx", pd.Series(dtype=float))
        atr = indicators.get("atr", pd.Series(dtype=float))
        vix_regime = indicators.get("vix_regime", pd.Series(dtype=object))

        # Volume 20-day rolling avg
        vol_avg20 = volume.rolling(20).mean()

        # ── Additive features (vectorized, in-memory) ─────────────────────────
        open_col = df["open"] if "open" in df.columns else close
        liq_class = df["liquidity_class"] if "liquidity_class" in df.columns else None
        vol_regimes = ml_features.compute_volatility_regime(
            close, atr=atr, liquidity_class=liq_class
        )
        vol_regime_enc = ml_features.encode_volatility_regime(vol_regimes)
        macro_sens = ml_features.compute_macro_sensitivity(
            close,
            open_=open_col,
            vix_regime=vix_regime if not vix_regime.empty else None,
            spx_futures_close=indicators.get("spx_futures_close"),
        )
        macro_flag = ml_features.macro_override_flag(
            df.index,
            close=close,
            open_=open_col,
            vix_regime=vix_regime if not vix_regime.empty else None,
            volatility_regime=vol_regimes,
        )
        overnight_weighted = ml_features.compute_overnight_return_weighted(open_col, close)

        for i, (ts, row) in enumerate(df.iterrows()):
            try:
                c = float(row["close"])
                if c <= 0:
                    continue

                # SMA ratio: SMA50 / SMA200 (> 1 = bullish cross)
                s50 = float(sma50.get(ts, np.nan)) if not sma50.empty else np.nan
                s200 = float(sma200.get(ts, np.nan)) if not sma200.empty else np.nan
                s20 = float(sma20.get(ts, np.nan)) if not sma20.empty else np.nan
                sma_ratio = (s50 / s200) if (s200 and s200 != 0 and not np.isnan(s200)) else 1.0

                macd_val = float(macd.get(ts, 0.0)) if not macd.empty else 0.0
                macd_sig_val = float(macd_signal.get(ts, 0.0)) if not macd_signal.empty else 0.0
                adx_val = float(adx.get(ts, 20.0)) if not adx.empty else 20.0
                atr_val = float(atr.get(ts, 0.0)) if not atr.empty else 0.0
                atr_norm = (atr_val / c) if c != 0 else 0.0

                # VIX regime encoding
                regime_str = str(vix_regime.get(ts, "NORMAL")) if not vix_regime.empty else "NORMAL"
                vix_enc = VIX_REGIME_MAP.get(regime_str, 1)

                # Recent returns
                if i >= 5:
                    c5 = float(df["close"].iloc[i - 5])
                    ret5 = (c - c5) / c5 if c5 != 0 else 0.0
                else:
                    ret5 = 0.0

                if i >= 10:
                    c10 = float(df["close"].iloc[i - 10])
                    ret10 = (c - c10) / c10 if c10 != 0 else 0.0
                else:
                    ret10 = 0.0

                if i >= 20:
                    c20 = float(df["close"].iloc[i - 20])
                    ret20 = (c - c20) / c20 if c20 != 0 else 0.0
                else:
                    ret20 = 0.0

                # Volume ratio
                vol_a = float(vol_avg20.iloc[i]) if i < len(vol_avg20) else 0.0
                vol_r = float(volume.iloc[i])
                vol_ratio = (vol_r / vol_a) if vol_a > 0 else 1.0

                # SMA20 distance
                sma20_dist = ((c - s20) / s20) if (s20 and s20 != 0 and not np.isnan(s20)) else 0.0

                feature_row = [
                    sma_ratio,
                    macd_val,
                    macd_sig_val,
                    adx_val,
                    vix_enc,
                    ret5,
                    ret10,
                    ret20,
                    vol_ratio,
                    atr_norm,
                    sma20_dist,
                    float(vol_regime_enc.iloc[i]),
                    float(macro_sens.iloc[i]),
                    float(macro_flag.iloc[i]),
                    float(overnight_weighted.iloc[i]),
                ]
                rows.append(feature_row)

                # Time-decay weight: Weight(t) = exp(-LAMBDA_LONG × age_in_days)
                ts_naive = ts
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    ts_naive = ts.replace(tzinfo=None)
                age_days = max(0.0, (now - ts_naive).days)
                w = math.exp(-settings.LAMBDA_LONG * age_days)
                weights.append(w)

            except Exception as exc:
                logger.debug("Skipping row at %s: %s", ts, exc)
                continue

        if not rows:
            return np.array([]).reshape(0, len(FEATURE_NAMES)), np.array([])

        return np.array(rows, dtype=np.float32), np.array(weights, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, indicators: dict) -> dict:
        """
        Train XGBoost on historical daily VOO data.

        Target:
          y(t) = 1 if Close(t+21) / Close(t) > 1.0 else 0

        Model hyper-parameters:
          n_estimators=200, max_depth=5, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
          eval_metric='logloss'

        Sample weight = time-decay weight per row.

        Saves to: ml/models/long_trend_model.pkl

        Returns:
            {"accuracy": float, "feature_importances": dict}
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            # Regular-hours only
            if "is_extended_hours" in df.columns:
                df = df[df["is_extended_hours"] == False].copy()

            # Build labels: 21-day forward return
            df = df.copy()
            df["future_close"] = df["close"].shift(-21)
            df.dropna(subset=["future_close"], inplace=True)
            df["label"] = (df["future_close"] > df["close"]).astype(int)

            # Trim indicators to match df
            trimmed_indicators = {
                k: v.reindex(df.index) if isinstance(v, pd.Series) else v
                for k, v in indicators.items()
            }

            X, weights = self.build_features(df, trimmed_indicators)
            y = df["label"].values[: len(X)]

            if len(X) < 50:
                logger.warning("Not enough data to train long-trend model (%d rows)", len(X))
                return {"accuracy": 0.0, "feature_importances": {}}

            # Normalize time-decay weights to mean 1.0. With a decade of
            # history the raw exp(-λ·age) weights shrink to ~1e-8, so the
            # summed hessian never reaches XGBoost's min_child_weight and the
            # trees degenerate to a constant prediction.
            mean_w = float(weights.mean())
            if mean_w > 0:
                weights = weights / mean_w

            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, weights, test_size=0.2, shuffle=False
            )

            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
            )
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))

            # ── Walk-forward evaluation + probability calibration ────────────
            # Honest out-of-sample metrics (purged chronological folds with a
            # 21-row embargo) plus a calibrator fitted on pooled OOS
            # predictions. Never blocks training on failure.
            calibration_summary: dict = {"calibrated": False}
            try:
                def _factory():
                    return xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        use_label_encoder=False,
                        random_state=42,
                    )

                wf_metrics, oos_probs, oos_labels = (
                    ml_calibration.walk_forward_evaluate(
                        X, y, weights, model_factory=_factory
                    )
                )
                calibration_summary.update(wf_metrics)
                calibrator = None
                if wf_metrics.get("evaluated"):
                    calibrator = ml_calibration.fit_calibrator(oos_probs, oos_labels)
                if calibrator is not None:
                    cal_brier = ml_calibration.calibrated_brier(
                        calibrator, oos_probs, oos_labels
                    )
                    calibration_summary["calibrated"] = True
                    calibration_summary["calibration_method"] = calibrator.method
                    calibration_summary["calibrated_brier_score"] = cal_brier
                    if ml_calibration.save_calibrator(calibrator):
                        self.calibrator = calibrator
                        self._calibrator_mtime = None  # force mtime re-read
                else:
                    calibration_summary.setdefault(
                        "reason", "calibrator could not be fitted"
                    )
                    logger.warning(
                        "ml_calibration_skipped model=long_trend reason=%s",
                        calibration_summary.get("reason"),
                    )
                ml_calibration.save_calibration_report(calibration_summary)
            except Exception as exc:
                logger.error("Long-trend calibration error: %s", exc)
                calibration_summary = {"calibrated": False, "reason": str(exc)}

            # Save model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            self.model = model
            self._model_loaded = True

            # Feature importances
            importances = dict(
                zip(FEATURE_NAMES, model.feature_importances_.tolist())
            )

            # Degeneracy health check: a broken train (e.g. vanishing sample
            # weights) can yield a constant predictor that still reports the
            # base-rate accuracy. Verify predictions actually vary.
            degenerate, degeneracy_reason = check_model_degeneracy(model, X)
            if degenerate:
                logger.error(
                    "ml_model_degenerate model=long_trend reason=%s",
                    degeneracy_reason,
                )

            logger.info("Long-trend model trained: accuracy=%.4f", acc)
            return {
                "accuracy": acc,
                "feature_importances": importances,
                "degenerate": degenerate,
                "degeneracy_reason": degeneracy_reason,
                "calibration": calibration_summary,
            }

        except Exception as exc:
            logger.error("Long-trend model training error: %s", exc)
            return {"accuracy": 0.0, "feature_importances": {}}

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> float:
        """
        Return BUY probability in [0, 1].

        Loads model from disk if not already loaded.

        Args:
            features: 1D or 2D ndarray of shape (n_features,) or (1, n_features)

        Returns:
            float in [0.0, 1.0]
        """
        try:
            self._maybe_reload()

            if self.model is None:
                logger.warning("Long-trend model not available; returning 0.5")
                self.last_prediction_was_fallback = True
                return 0.5

            if features.ndim == 1:
                features = features.reshape(1, -1)

            prob = float(self.model.predict_proba(features)[0][1])

            # Apply the persisted probability calibrator when available so the
            # gauge consumes an honest confidence. Raw probability is the
            # fallback — calibration must never break prediction.
            if self.calibrator is not None:
                try:
                    prob = self.calibrator.transform(prob)
                except Exception as exc:
                    logger.error("Long-trend calibration apply error: %s", exc)

            self.last_prediction_was_fallback = False
            return float(prob)
        except Exception as exc:
            logger.error("Long-trend predict error: %s", exc)
            self.last_prediction_was_fallback = True
            return 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # Model persistence
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        """
        Load model from ml/models/long_trend_model.pkl if it exists.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            if MODEL_PATH.exists():
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
                # Guard: a model pickled before the feature-set extension has
                # a smaller feature count. Never crash prediction on it —
                # discard, log, and fall back to neutral until retraining.
                n_in = getattr(model, "n_features_in_", None)
                if n_in is not None and int(n_in) != len(FEATURE_NAMES):
                    logger.warning(
                        "ml_model_stale model=long_trend expected_features=%d "
                        "found=%d action=discard_await_retrain",
                        len(FEATURE_NAMES), int(n_in),
                    )
                    self.model = None
                    self._model_loaded = True
                    return False
                self.model = model
                self._model_loaded = True
                logger.info("Long-trend model loaded from %s", MODEL_PATH)
                return True
            else:
                logger.info("No long-trend model file found at %s", MODEL_PATH)
                self._model_loaded = True  # Prevent repeated load attempts
                return False
        except Exception as exc:
            logger.error("Error loading long-trend model: %s", exc)
            self._model_loaded = True
            return False

    def is_neutral_fallback(self) -> bool:
        """
        True when the model is unavailable (missing, stale, or failed to load)
        and predict() would return the neutral 0.5 fallback.
        """
        self._maybe_reload()
        return self.model is None

    def build_latest_features(
        self,
        df: pd.DataFrame,
        indicators: dict,
    ) -> Optional[np.ndarray]:
        """
        Build the single feature vector for the most-recent data point.

        Returns:
            np.ndarray of shape (1, n_features) or None if insufficient data.
        """
        try:
            X, _ = self.build_features(df, indicators)
            if len(X) == 0:
                return None
            return X[-1:].astype(np.float32)
        except Exception as exc:
            logger.error("build_latest_features error: %s", exc)
            return None
