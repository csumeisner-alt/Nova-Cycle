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

Target (shared rally-event definition — see rally_event.py):
  y = 1 if max(close[t+1 : t+12]) / close[t] - 1 > 0.003 else 0
  i.e. the price reaches +0.3% at ANY point within the next hour, matching
  the missed-rally detector and the alert the user actually wants.

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
from ml import calibration as ml_calibration
from ml.model_health import check_model_degeneracy
from rally_event import RALLY_HORIZON_BARS, rally_event_labels

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

# Label horizon in 5-min bars: the label covers the NEXT 12 bars (1 hour).
# Walk-forward evaluation must purge at least this many rows between each
# training window and its test window, or training labels leak test-window
# prices.  Single source of truth: rally_event.RALLY_HORIZON_BARS.
LABEL_HORIZON_BARS = RALLY_HORIZON_BARS

# ─────────────────────────────────────────────────────────────────────────────
# Leakage audit (enforced by tests/test_short_leakage.py)
# ─────────────────────────────────────────────────────────────────────────────
# Every feature must be computable from data at or before bar t (causal),
# because the label covers bars (t, t+12]:
#   - rsi / stoch* / bollinger / atr*        → rolling windows ENDING at t
#   - return_1h/2h/4h/1d                     → close[t] vs close[t-n] (backward)
#   - return_overnight                        → open[t] vs close[t-1]
#   - volume_ratio                            → volume[t] vs 20-bar avg ending t
#   - is_extended / gap_percent / liquidity_score → known at bar t
#   - additive features (volatility regime, macro sensitivity/override,
#     gap momentum, liquidity compression)    → rolling/backward only
# Historical bug this guards against: the 85/15 validation split was cut at a
# single boundary with no purge gap, and the feature scaler was fitted on ALL
# rows (train + validation), so reported validation accuracy (~98%) was
# leakage-inflated while live confidence collapsed to ~0.
# ─────────────────────────────────────────────────────────────────────────────


class ScaledMLP:
    """StandardScaler + MLPClassifier pipeline used for walk-forward folds
    and the final fit.

    Fitting the scaler INSIDE each training window (instead of on the full
    dataset) removes the scaler-statistics leakage of the previous
    implementation. sample_weight is approximated by oversampling the
    highest-weight (most recent) quartile, since MLPClassifier.fit does not
    support sample weights.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
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

    def fit(self, X, y, sample_weight=None, verbose=False):
        Xs = self.scaler.fit_transform(X)
        X_fit, y_fit = Xs, y
        try:
            if sample_weight is not None and len(sample_weight) == len(Xs):
                threshold = np.percentile(sample_weight, 75) if len(sample_weight) else 0.0
                recent_mask = sample_weight >= threshold
                if 0 < recent_mask.sum() < len(Xs):
                    X_fit = np.vstack([Xs, Xs[recent_mask]])
                    y_fit = np.concatenate([y, y[recent_mask]])
        except Exception:
            X_fit, y_fit = Xs, y
        # Class balancing: the ">0.3% move in 1h" label is rare (~5-10% base
        # rate), and the unbalanced MLP collapsed its probabilities to ~1e-6
        # for every live bar (a pinned −40 gauge contribution). Oversample the
        # minority class up to parity so the network learns a discriminative
        # boundary instead of the majority prior.
        try:
            X_fit, y_fit = self._balance_classes(X_fit, y_fit)
        except Exception:
            pass
        self.mlp.fit(X_fit, y_fit)
        return self

    @staticmethod
    def _balance_classes(X, y):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            return X, y
        minority = classes[np.argmin(counts)]
        n_min, n_maj = counts.min(), counts.max()
        if n_min == 0 or n_min == n_maj:
            return X, y
        rng = np.random.default_rng(42)
        idx_min = np.where(y == minority)[0]
        extra = rng.choice(idx_min, size=int(n_maj - n_min), replace=True)
        return np.vstack([X, X[extra]]), np.concatenate([y, y[extra]])

    def predict_proba(self, X):
        return self.mlp.predict_proba(self.scaler.transform(X))


class ShortTrendModel:
    """scikit-learn MLP predicting 1-hour forward return > 0.3%."""

    def __init__(self):
        self.model: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.calibrator: Optional[ml_calibration.ProbabilityCalibrator] = None
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
            cal_path = ml_calibration.calibrator_path("short_trend")
            cal_mtime = cal_path.stat().st_mtime if cal_path.exists() else None
        except OSError:
            cal_mtime = None
        if cal_mtime != self._calibrator_mtime:
            calibrator = ml_calibration.load_calibrator("short_trend")
            self.calibrator = calibrator
            # Only pin the mtime when the load succeeded (or the file is
            # genuinely absent); a transient read failure must be retried on
            # the next prediction rather than silently disabling calibration.
            if calibrator is not None or cal_mtime is None:
                self._calibrator_mtime = cal_mtime

        # The short label is a rare event, so its calibrated neutral point is
        # the observed OOS positive rate rather than 0.5.  Keep this metadata
        # separate from the calibrator pickle so a report can be inspected and
        # safely reloaded without changing model persistence format.
        try:
            report_path = ml_calibration.calibration_report_path("short_trend")
            report_mtime = report_path.stat().st_mtime if report_path.exists() else None
        except OSError:
            report_mtime = None
        if report_mtime != self._calibration_report_mtime:
            report = ml_calibration.get_calibration_report("short_trend")
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
        liquidity_score: float = 1.0,
        gap_percent: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build feature matrix from 5-min candle DataFrame.

        Extended-hours weight:
          base_weight = 0.5 if is_extended_hours else 1.0

        Time-decay:
          Weight = base_weight × exp(-LAMBDA_SHORT × age_in_minutes)

        Returns:
            (X: np.ndarray [n, N_FEATURES], weights: np.ndarray [n],
             valid_pos: np.ndarray [n] — positional indices into df of the
             rows that produced a feature row.  Callers building labels MUST
             index labels with valid_pos rather than slicing positionally,
             or a skipped mid-series row silently misaligns X and y.)
        """
        rows = []
        weights = []
        valid_pos = []

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
                valid_pos.append(i)

                base_w = 0.5 if is_ext else 1.0
                age_min = max(0.0, (now - ts_naive).total_seconds() / 60.0)
                w = base_w * math.exp(-settings.LAMBDA_SHORT * age_min)
                weights.append(max(w, 1e-6))

            except Exception as exc:
                logger.debug("Skipping 5-min row %s: %s", ts, exc)
                continue

        if not rows:
            return (
                np.array([]).reshape(0, N_FEATURES),
                np.array([]),
                np.array([], dtype=np.int64),
            )

        return (
            np.array(rows, dtype=np.float32),
            np.array(weights, dtype=np.float32),
            np.array(valid_pos, dtype=np.int64),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, indicators: dict) -> dict:
        """
        Train the MLPClassifier.

        Target (shared rally-event definition, rally_event.py):
          y = 1 if max(Close[t+1 : t+12]) / Close[t] - 1 > 0.003 else 0

        Saves scaler + model to ml/models/short_trend_model.pkl.

        Returns:
            {"accuracy": float, "val_accuracy": float}
        """
        try:
            df = df.copy()
            df["label"] = rally_event_labels(df["close"])
            df.dropna(subset=["label"], inplace=True)
            df["label"] = df["label"].astype(int)

            X, sample_weights, valid_pos = self.build_features(df, indicators)
            # Index labels by the positions that actually produced feature
            # rows — a positional slice would misalign X and y whenever
            # build_features skips a mid-series row.
            y = df["label"].values[valid_pos]

            if len(X) < 100:
                logger.warning(
                    "Not enough 5-min data to train short-trend model (%d rows)", len(X)
                )
                return {"accuracy": 0.0, "val_accuracy": 0.0}

            # ── VIX regime labels aligned to feature rows ────────────────────
            # Passed to walk_forward_evaluate so the report includes per-regime
            # OOS metrics (same as long-trend) rather than overall metrics only.
            #
            # indicators["vix_regime"] is a string Series from TechnicalIndicators
            # (values: "LOW", "NORMAL", "HIGH", "EXTREME").  Convert to the
            # same integer encoding that _regime_breakdown expects:
            #   LOW=0, NORMAL=1, HIGH=2, EXTREME=3
            # Any unknown or missing value defaults to NORMAL (1).
            _VIX_REGIME_CODE = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}
            vix_regime_for_wf = None
            vix_regime_raw = indicators.get("vix_regime")
            if vix_regime_raw is not None:
                try:
                    if isinstance(vix_regime_raw, pd.Series) and not vix_regime_raw.empty:
                        # Align the Series to df's index so positional slicing
                        # matches how y is constructed above (y = df["label"][:len(X)]).
                        vr_aligned = (
                            vix_regime_raw
                            .reindex(df.index)
                            .fillna("NORMAL")
                            .astype(str)
                            .str.upper()
                            .map(_VIX_REGIME_CODE)
                            .fillna(1)  # unknown strings → NORMAL
                            .astype(np.int32)
                        )
                        vix_regime_for_wf = vr_aligned.values[valid_pos]
                    elif isinstance(vix_regime_raw, str):
                        # Scalar string regime: broadcast to all rows
                        code = _VIX_REGIME_CODE.get(str(vix_regime_raw).upper(), 1)
                        vix_regime_for_wf = np.full(len(X), code, dtype=np.int32)
                    elif isinstance(vix_regime_raw, (int, float, np.integer)):
                        # Already numeric (unlikely but guard): broadcast directly
                        vix_regime_for_wf = np.full(len(X), int(vix_regime_raw), dtype=np.int32)
                except Exception as exc:
                    logger.warning(
                        "short_trend: could not extract vix_regime for walk-forward: %s", exc
                    )
                    vix_regime_for_wf = None

            # ── Purged walk-forward evaluation (honest OOS metrics) ──────────
            # Chronological folds with an embargo gap >= the 12-bar label
            # horizon so no training label overlaps test-window prices; the
            # scaler is re-fitted inside each fold's training window.
            wf_metrics, oos_probs, oos_labels = ml_calibration.walk_forward_evaluate(
                X, y, sample_weights,
                model_factory=ScaledMLP,
                embargo=LABEL_HORIZON_BARS,
                regime_labels=vix_regime_for_wf,
            )
            ml_calibration.save_walkforward_report("short_trend", wf_metrics)
            oos_acc = wf_metrics.get("oos_accuracy") if wf_metrics.get("evaluated") else None

            # ── Probability calibration on pooled OOS predictions ────────────
            # The MLP trains on a class-balanced sample, so its raw probability
            # does not equal the true base rate of a >0.3% move within the
            # hour. Fit a calibrator on the pooled out-of-sample predictions
            # (same machinery as the long-trend model) so the gauge's ML
            # contribution reflects real-world frequencies. Never blocks
            # training on failure.
            calibration_summary: dict = {"calibrated": False}
            try:
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
                    if ml_calibration.save_calibrator(calibrator, "short_trend"):
                        self.calibrator = calibrator
                        self._calibrator_mtime = None  # force mtime re-read
                else:
                    calibration_summary.setdefault(
                        "reason", "calibrator could not be fitted"
                    )
                    logger.warning(
                        "ml_calibration_skipped model=short_trend reason=%s",
                        calibration_summary.get("reason"),
                    )
                ml_calibration.save_calibration_report(
                    calibration_summary, "short_trend"
                )
                positive_rate = calibration_summary.get("positive_rate")
                if positive_rate is not None:
                    try:
                        self.calibration_base_rate = min(
                            0.99, max(0.01, float(positive_rate))
                        )
                    except (TypeError, ValueError):
                        self.calibration_base_rate = None
            except Exception as exc:
                logger.error("Short-trend calibration error: %s", exc)
                calibration_summary = {"calibrated": False, "reason": str(exc)}

            # ── Final fit on ALL data (scaler fitted here, on training data
            # only — there is no held-out set for the deployed model) ─────────
            pipeline = ScaledMLP()
            pipeline.fit(X, y, sample_weight=sample_weights)
            model, scaler = pipeline.mlp, pipeline.scaler
            X_scaled = scaler.transform(X)

            train_acc = float(model.score(X_scaled, y))
            # Honest reported accuracy: purged walk-forward OOS when available.
            val_acc = float(oos_acc) if oos_acc is not None else 0.0
            reported_acc = float(oos_acc) if oos_acc is not None else train_acc

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
                "Short-trend MLP trained: train_acc=%.4f  oos_acc=%s",
                train_acc,
                f"{oos_acc:.4f}" if oos_acc is not None else "n/a",
            )
            return {
                # Honest headline accuracy: purged walk-forward OOS when the
                # evaluation ran (train accuracy on an MLP is near-memorized
                # and was the source of the misleading 98.6% number).
                "accuracy": reported_acc,
                "accuracy_metric": (
                    "purged_walk_forward_oos" if oos_acc is not None else "train"
                ),
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "walk_forward": wf_metrics,
                "calibration": calibration_summary,
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
            self._maybe_reload()

            if self.model is None:
                logger.warning("Short-trend model not available; returning 0.5")
                self.last_prediction_was_fallback = True
                return 0.5

            if features.ndim == 1:
                features = features.reshape(1, -1)

            if self.scaler is not None:
                features = self.scaler.transform(features)

            probs = self.model.predict_proba(features)
            # predict_proba returns [[p_class0, p_class1]]
            prob = float(probs[0][1]) if probs.shape[1] > 1 else float(probs[0][0])

            # Apply the persisted probability calibrator when available: the
            # model trains on a class-balanced sample, so the raw probability
            # overstates the true chance of a >0.3% move within the hour. Raw
            # probability is the fallback — calibration must never break
            # prediction.
            if self.calibrator is not None:
                try:
                    prob = self.calibrator.transform(prob)
                except Exception as exc:
                    logger.error("Short-trend calibration apply error: %s", exc)

            self.last_prediction_was_fallback = False
            return float(prob)

        except Exception as exc:
            logger.error("Short-trend predict error: %s", exc)
            self.last_prediction_was_fallback = True
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
        self._maybe_reload()
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
            X, _, _ = self.build_features(df, indicators, liquidity_score, gap_percent)
            if len(X) == 0:
                return None
            return X[-1:].astype(np.float32)
        except Exception as exc:
            logger.error("build_latest_features (short) error: %s", exc)
            return None
