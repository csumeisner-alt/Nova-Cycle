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
from ml.training_status import get_last_successful_accuracy_metric

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "long_trend_model.pkl"
# Sidecar JSON that records the target_type the active pkl was trained for.
# Written on every successful promotion; read during load_model() to detect
# mismatches when LONG_TARGET_TYPE changes between retrains.
_META_PATH = MODEL_DIR / "long_trend_meta.json"

VIX_REGIME_MAP = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}

# ── Feature name registry ─────────────────────────────────────────────────────
# _BASE_FEATURE_NAMES: the stable 19-feature set used by all trained models.
# _BROADER_CONTEXT_FEATURE_NAMES: 8 additional features (4 values + 4 freshness
#   flags) enabled only when settings.LONG_BROADER_CONTEXT_ENABLED=True.
# FEATURE_NAMES: the authoritative list consumed by build_features(), train(),
#   and load_model().  Computed once at import time from the settings singleton.

_BASE_FEATURE_NAMES = [
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
    # Raw VIX context avoids collapsing materially different NORMAL readings
    # into the same categorical value.
    "vix_level_norm",
    "vix_change_5d",
    "vix_percentile_1y",
    "vix_missing",
]

# Broader market context features — added only after OOS viability is confirmed.
# Paired as (value, freshness_flag): missing=1.0 when the external data source
# is absent or stale so the model can learn to ignore unavailable context rather
# than rely on a hard-coded neutral value for signal.
_BROADER_CONTEXT_FEATURE_NAMES = [
    "vix_term_slope",        # VIX9D/VIX3M − 1 (negative=contango, positive=backwardation)
    "vix_term_missing",      # 1.0 when term-structure data absent (proxy active)
    "credit_stress_score",   # HY-IG spread proxy [0,1]; 0.5=neutral, >0.5=stress
    "credit_stress_missing", # 1.0 when HYG/LQD data absent
    "breadth_score",         # NYSE A/D momentum [0,1]; >0.5=improving breadth
    "breadth_missing",       # 1.0 when NYAD data absent
    "rates_level_norm",      # 10Y yield / 8% cap → [0,1]
    "rates_missing",         # 1.0 when TNX data absent
]

FEATURE_NAMES: list[str] = _BASE_FEATURE_NAMES + (
    _BROADER_CONTEXT_FEATURE_NAMES if settings.LONG_BROADER_CONTEXT_ENABLED else []
)


def _walk_forward_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray],
    model_factory,
    n_splits: int = 5,
    embargo: int = 21,
) -> dict:
    """Purged chronological walk-forward for a three-state classifier.

    Returns a metrics dict with ``evaluated``, ``macro_f1``,
    ``oos_balanced_accuracy``, ``per_class``, and ``folds``.
    Uses the same split logic as the binary walk_forward_evaluate so results
    are directly comparable between target types.
    """
    try:
        from sklearn.metrics import (
            f1_score as _f1,
            precision_recall_fscore_support as _prf,
            balanced_accuracy_score as _bal_acc,
        )
    except ImportError:
        return {"evaluated": False, "reason": "sklearn not available"}

    n = len(X)
    min_train = max(100, embargo * 3)
    if n < min_train + embargo + n_splits:
        return {
            "evaluated": False,
            "reason": f"not enough rows ({n}) for multiclass walk-forward",
        }

    test_start = max(min_train + embargo, int(n * 0.5))
    fold_edges = np.linspace(test_start, n, n_splits + 1, dtype=int)

    oos_preds: list = []
    oos_labels: list = []
    fold_stats = []

    for k in range(n_splits):
        t0, t1 = int(fold_edges[k]), int(fold_edges[k + 1])
        if t1 <= t0:
            continue
        train_end = t0 - embargo
        if train_end < min_train:
            continue
        model = model_factory()
        w = None
        if weights is not None and len(weights) == n:
            w = weights[:train_end].copy()
            mw = float(w.mean())
            if mw > 0:
                w = w / mw
        model.fit(X[:train_end], y[:train_end], sample_weight=w, verbose=False)
        preds = model.predict(X[t0:t1])
        oos_preds.append(preds)
        oos_labels.append(y[t0:t1])
        fold_acc = float((preds == y[t0:t1]).mean())
        fold_stats.append({
            "fold": k + 1,
            "train_rows": int(train_end),
            "test_rows": int(t1 - t0),
            "accuracy": fold_acc,
        })

    if not oos_preds:
        return {"evaluated": False, "reason": "no valid walk-forward folds"}

    all_preds = np.concatenate(oos_preds)
    all_labels = np.concatenate(oos_labels).astype(int)

    classes = sorted(np.unique(np.concatenate([all_labels, all_preds])).tolist())
    macro_f1 = float(_f1(all_labels, all_preds, average="macro", zero_division=0))
    bal_acc = float(_bal_acc(all_labels, all_preds))
    overall_acc = float((all_preds == all_labels).mean())

    prec, rec, f1_scores, support = _prf(
        all_labels, all_preds, labels=classes, average=None, zero_division=0
    )
    class_names = {0: "risk_off", 1: "neutral", 2: "risk_on"}
    per_class = []
    for i, cls in enumerate(classes):
        per_class.append({
            "class": int(cls),
            "name": class_names.get(cls, str(cls)),
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1_scores[i]), 4),
            "support": int(support[i]),
        })

    return {
        "evaluated": True,
        "method": "purged_walk_forward_multiclass",
        "n_splits": len(fold_stats),
        "embargo_rows": int(embargo),
        "oos_samples": int(len(all_labels)),
        "oos_accuracy": round(overall_acc, 4),
        "oos_balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "folds": fold_stats,
    }


class LongTrendModel:
    """XGBoost model predicting 21-day forward return direction for VOO."""

    def __init__(self):
        self.model = None
        self._model_feature_count: Optional[int] = None
        self.calibrator = None
        self._model_loaded = False
        self._loaded_mtime: Optional[float] = None
        self._calibrator_mtime: Optional[float] = None
        self.calibration_base_rate: Optional[float] = None
        self._calibration_report_mtime: Optional[float] = None
        # True when no gate-passing trained model is available; the long signal
        # is served from a calibrated majority-class base rate instead.
        self._baseline_mode: bool = False
        # Set by predict() on each call; lets predict_long detect a silent 0.5.
        self.last_prediction_was_fallback: bool = False
        # Target type reported by the last successful promotion (loaded from
        # the meta sidecar on each load_model() call).  Falls back to
        # settings.LONG_TARGET_TYPE when no sidecar exists.
        self._promoted_target_type: Optional[str] = None
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def target_type(self) -> str:
        """Target type the model was promoted for (direction | drawdown_event | three_state).

        Reads the value recorded by the last successful training run from the
        meta sidecar.  Falls back to settings.LONG_TARGET_TYPE when the sidecar
        is absent so a freshly deployed instance with no trained model uses the
        configured target.
        """
        return self._promoted_target_type or settings.LONG_TARGET_TYPE

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
        return self._target_aware_base_rate() or 0.5

    def is_baseline_mode(self) -> bool:
        """True when no gate-passing trained model is available.

        The long signal falls back to a calibrated majority-class base rate
        (get_baseline_probability()) instead of a trained-model prediction.
        Baseline mode is set when:
          - the on-disk pkl is the legacy 15-feature model (OOS lift ≈ −29 pp),
          - no pkl file exists on disk, or
          - the pkl failed to load.
        Baseline mode clears automatically when a gate-passing 19-feature model
        is promoted to disk and detected by the mtime-based reload path.
        """
        self._maybe_reload()
        return self._baseline_mode

    def get_baseline_probability(self) -> float:
        """Return the target-appropriate base rate as a directional ml_confidence.

        For direction models: calibration report ``positive_rate`` = historical
        P(BUY label) ≈ 0.73 for VOO/21d/2% — mildly bullish, reflects the
        equity risk premium.

        For drawdown_event models: ``1 − positive_rate`` = historical
        P(no drawdown) — typically high (≈0.92 for a 5% threshold), which is the
        correct majority-class prediction in "no drawdown" space.

        For three_state models: 0.5 (neutral), since no binary base rate applies.

        Sourced from the calibration report at runtime; never hardcoded.
        Falls back to 0.5 when no report is available.
        """
        self._maybe_reload()
        return self._target_aware_base_rate() or 0.5

    def _target_aware_base_rate(self) -> Optional[float]:
        """Return a valid (0,1)-bounded base rate adjusted for the active target type.

        Returns None when no calibration base rate has been loaded yet.
        """
        rate = self.calibration_base_rate  # positive_rate from calibration report
        if rate is None:
            return None
        target = self.target_type
        if target == "drawdown_event":
            # positive_rate is drawdown event prevalence; ml_confidence semantics
            # are inverted (high = no drawdown = bullish).
            return min(0.99, max(0.01, 1.0 - rate))
        elif target == "three_state":
            # No meaningful binary base rate; return neutral.
            return 0.5
        else:
            # direction: positive_rate directly represents bullish probability.
            return rate

    # ──────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────────────

    def build_features(
        self,
        df: pd.DataFrame,
        indicators: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the feature matrix from regular-hours daily data.

        Time-decay:
          Weight(t) = exp(-LAMBDA_LONG × age_in_days)
          where age_in_days = (today - timestamp).days

        Args:
            df:         VOO daily DataFrame (regular-hours only; indexed by timestamp)
            indicators: Output of TechnicalIndicators.compute_all()

        Returns:
            (X: np.ndarray [n, features], weights: np.ndarray [n],
             valid_positions: np.ndarray [n] of integer row positions in df
             that contributed to X, enabling caller to align labels by
             position rather than assuming no rows were skipped.)
        """
        rows = []
        weights = []
        valid_positions = []

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
        vix_level = indicators.get("vix_level", pd.Series(dtype=float))
        vix_change_5d = indicators.get("vix_change_5d", pd.Series(dtype=float))
        vix_percentile_1y = indicators.get(
            "vix_percentile_1y", pd.Series(dtype=float)
        )
        vix_missing = indicators.get("vix_missing", pd.Series(dtype=bool))

        # Volume 20-day rolling avg.
        # When pre-computed temporal columns are present (set by train() before
        # the meaningful-move filter is applied) use them directly so the
        # inference path (iloc-based) and the training path (filtered subset)
        # both see the same vol_avg20 value for a given timestamp.
        if "_vol_avg20" in df.columns:
            vol_avg20 = df["_vol_avg20"]
        else:
            vol_avg20 = volume.rolling(20).mean()

        # ── Additive features (vectorized, in-memory) ─────────────────────────
        # When pre-computed columns are present (set by train() on the FULL,
        # UNFILTERED df before the meaningful-move filter) use them directly.
        # Recomputing these rolling-window features on the filtered subset
        # (non-contiguous dates) makes their windows span filtered rows rather
        # than real trading days, creating a train/inference feature mismatch —
        # the same bug class as the _return_* fix in train().
        open_col = df["open"] if "open" in df.columns else close
        liq_class = df["liquidity_class"] if "liquidity_class" in df.columns else None
        if "_vol_regime_enc" in df.columns:
            vol_regime_enc = df["_vol_regime_enc"]
        else:
            vol_regimes = ml_features.compute_volatility_regime(
                close, atr=atr, liquidity_class=liq_class
            )
            vol_regime_enc = ml_features.encode_volatility_regime(vol_regimes)
        if "_macro_sens" in df.columns:
            macro_sens = df["_macro_sens"]
        else:
            macro_sens = ml_features.compute_macro_sensitivity(
                close,
                open_=open_col,
                vix_regime=vix_regime if not vix_regime.empty else None,
                spx_futures_close=indicators.get("spx_futures_close"),
            )
        if "_macro_flag" in df.columns:
            macro_flag = df["_macro_flag"]
        else:
            if "_vol_regime_enc" in df.columns:
                # Recompute regimes only if needed for the flag fallback.
                vol_regimes = ml_features.compute_volatility_regime(
                    close, atr=atr, liquidity_class=liq_class
                )
            macro_flag = ml_features.macro_override_flag(
                df.index,
                close=close,
                open_=open_col,
                vix_regime=vix_regime if not vix_regime.empty else None,
                volatility_regime=vol_regimes,
            )
        if "_overnight_w" in df.columns:
            overnight_weighted = df["_overnight_w"]
        else:
            overnight_weighted = ml_features.compute_overnight_return_weighted(open_col, close)

        # ── Broader market context (gated by LONG_BROADER_CONTEXT_ENABLED) ────
        # Train path: pre-computed df columns survive the meaningful-move filter
        #   and are consumed directly (same value as on the full unfiltered df).
        # Inference path: compute vectorially from indicators on the spot.
        # All functions return (value_series, missing_series) and never raise.
        _ctx_vix_term_slope = None
        _ctx_vix_term_missing = None
        _ctx_credit_stress = None
        _ctx_credit_missing = None
        _ctx_breadth = None
        _ctx_breadth_missing = None
        _ctx_rates = None
        _ctx_rates_missing = None

        if settings.LONG_BROADER_CONTEXT_ENABLED:
            _stale_days = int(
                getattr(settings, "LONG_CONTEXT_STALENESS_MAX_DAYS", 5)
            )
            # VIX term structure: use the raw VIX level series as proxy base
            _vix_for_term = vix_level if not vix_level.empty else close
            if "_vix_term_slope" in df.columns:
                _ctx_vix_term_slope = df["_vix_term_slope"]
                _ctx_vix_term_missing = df["_vix_term_missing"]
            else:
                _ctx_vix_term_slope, _ctx_vix_term_missing = (
                    ml_features.compute_vix_term_structure(
                        _vix_for_term,
                        vix_short_close=indicators.get("vix_short_close"),
                        vix_long_close=indicators.get("vix_long_close"),
                        staleness_max_days=_stale_days,
                    )
                )
            if "_credit_stress_score" in df.columns:
                _ctx_credit_stress = df["_credit_stress_score"]
                _ctx_credit_missing = df["_credit_stress_missing"]
            else:
                _ctx_credit_stress, _ctx_credit_missing = (
                    ml_features.compute_credit_stress(
                        df.index,
                        hy_close=indicators.get("credit_hy_close"),
                        ig_close=indicators.get("credit_ig_close"),
                        staleness_max_days=_stale_days,
                    )
                )
            if "_breadth_score" in df.columns:
                _ctx_breadth = df["_breadth_score"]
                _ctx_breadth_missing = df["_breadth_missing"]
            else:
                _ctx_breadth, _ctx_breadth_missing = (
                    ml_features.compute_market_breadth(
                        df.index,
                        breadth_close=indicators.get("breadth_close"),
                        staleness_max_days=_stale_days,
                    )
                )
            if "_rates_level_norm" in df.columns:
                _ctx_rates = df["_rates_level_norm"]
                _ctx_rates_missing = df["_rates_missing"]
            else:
                _ctx_rates, _ctx_rates_missing = (
                    ml_features.compute_rates_level(
                        df.index,
                        rates_close=indicators.get("rates_close"),
                        staleness_max_days=_stale_days,
                    )
                )

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
                vix_val = (
                    float(vix_level.get(ts, np.nan))
                    if not vix_level.empty
                    else np.nan
                )
                vix_norm = (
                    np.clip(vix_val / 40.0, 0.0, 3.0)
                    if np.isfinite(vix_val)
                    else 0.5
                )
                vix_chg = (
                    float(vix_change_5d.get(ts, 0.0))
                    if not vix_change_5d.empty
                    else 0.0
                )
                vix_pct = (
                    float(vix_percentile_1y.get(ts, 0.5))
                    if not vix_percentile_1y.empty
                    else 0.5
                )
                vix_miss = (
                    float(bool(vix_missing.get(ts, True)))
                    if not vix_missing.empty
                    else 1.0
                )

                # Recent returns.
                # Use pre-computed columns (training path: values come from the
                # full, unfiltered df so iloc offsets correctly span real trading
                # days).  Fall back to iloc-based computation on the inference
                # path (build_latest_features calls this on the full series,
                # where i-5 really is 5 trading days ago).
                if "_return_5d" in df.columns:
                    _v = df["_return_5d"].iloc[i]
                    ret5 = float(_v) if pd.notna(_v) else 0.0
                elif i >= 5:
                    c5 = float(df["close"].iloc[i - 5])
                    ret5 = (c - c5) / c5 if c5 != 0 else 0.0
                else:
                    ret5 = 0.0

                if "_return_10d" in df.columns:
                    _v = df["_return_10d"].iloc[i]
                    ret10 = float(_v) if pd.notna(_v) else 0.0
                elif i >= 10:
                    c10 = float(df["close"].iloc[i - 10])
                    ret10 = (c - c10) / c10 if c10 != 0 else 0.0
                else:
                    ret10 = 0.0

                if "_return_20d" in df.columns:
                    _v = df["_return_20d"].iloc[i]
                    ret20 = float(_v) if pd.notna(_v) else 0.0
                elif i >= 20:
                    c20 = float(df["close"].iloc[i - 20])
                    ret20 = (c - c20) / c20 if c20 != 0 else 0.0
                else:
                    ret20 = 0.0

                # Volume ratio
                if "_vol_avg20" in df.columns:
                    _va = df["_vol_avg20"].iloc[i]
                    vol_a = float(_va) if pd.notna(_va) else 0.0
                else:
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
                    vix_norm,
                    vix_chg,
                    vix_pct,
                    vix_miss,
                ]
                # ── Broader context (appended after the 19-feature base) ──────
                if settings.LONG_BROADER_CONTEXT_ENABLED:
                    try:
                        feature_row.extend([
                            float(_ctx_vix_term_slope.iloc[i]),
                            float(_ctx_vix_term_missing.iloc[i]),
                            float(_ctx_credit_stress.iloc[i]),
                            float(_ctx_credit_missing.iloc[i]),
                            float(_ctx_breadth.iloc[i]),
                            float(_ctx_breadth_missing.iloc[i]),
                            float(_ctx_rates.iloc[i]),
                            float(_ctx_rates_missing.iloc[i]),
                        ])
                    except Exception:
                        # Data absent or index out of range: neutral + all missing
                        feature_row.extend([0.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0])
                rows.append(feature_row)
                valid_positions.append(i)

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
            return (
                np.array([]).reshape(0, len(FEATURE_NAMES)),
                np.array([]),
                np.array([], dtype=np.intp),
            )

        return (
            np.array(rows, dtype=np.float32),
            np.array(weights, dtype=np.float32),
            np.array(valid_positions, dtype=np.intp),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, indicators: dict) -> dict:
        """
        Train XGBoost on historical daily VOO data.

        Target:
          y(t) = 1 for a meaningful forward move at or above the configured
          threshold, 0 for a meaningful move at or below the negative
          threshold.  Near-flat outcomes are excluded as noise.

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

            # Snapshot dataset dimensions for the calibration report *before*
            # any row filtering so the report always describes what the caller
            # provided, not the post-filter subset.  _ds_labeled_rows is
            # updated after the meaningful-move filter below.
            _ds_total_candles = len(df)
            try:
                _ds_date_start = str(df.index[0].date()) if len(df) > 0 else None
                _ds_date_end   = str(df.index[-1].date()) if len(df) > 0 else None
            except Exception:
                _ds_date_start = _ds_date_end = None
            _ds_labeled_rows: int = 0  # updated after meaningful-move filter

            # Pre-compute temporally-correct return and volume features on the
            # FULL, UNFILTERED df BEFORE the meaningful-move filter is applied.
            # This is the root-cause fix for repeated OOS quality gate failures:
            # without this, build_features() falls back to iloc-based lookbacks
            # on the filtered subset (which is non-contiguous after removing
            # noise rows), so `iloc[i-5]` spans the 5th *labeled* row ago rather
            # than 5 *trading days* ago.  That can invert the sign of the return
            # features relative to the inference path (where the full unfiltered
            # df is passed), creating a train/inference mismatch and teaching the
            # model the wrong direction.  The _return_* and _vol_avg20 columns
            # survive the meaningful-move filter and are consumed by
            # build_features() when present.
            df = df.copy()
            df["_return_5d"]  = df["close"].pct_change(5)
            df["_return_10d"] = df["close"].pct_change(10)
            df["_return_20d"] = df["close"].pct_change(20)
            if "volume" in df.columns:
                df["_vol_avg20"] = df["volume"].rolling(20).mean()
            else:
                df["_vol_avg20"] = 0.0

            # Same treatment for the four additive rolling-window features:
            # compute them on the full unfiltered df so training sees the same
            # values as inference (where the full df is always passed).
            _close_full = df["close"]
            _open_full = df["open"] if "open" in df.columns else _close_full
            _liq_full = (
                df["liquidity_class"] if "liquidity_class" in df.columns else None
            )
            _atr_full = indicators.get("atr", pd.Series(dtype=float))
            _vix_regime_full = indicators.get("vix_regime", pd.Series(dtype=object))
            _vol_regimes_full = ml_features.compute_volatility_regime(
                _close_full, atr=_atr_full, liquidity_class=_liq_full
            )
            df["_vol_regime_enc"] = ml_features.encode_volatility_regime(
                _vol_regimes_full
            )
            df["_macro_sens"] = ml_features.compute_macro_sensitivity(
                _close_full,
                open_=_open_full,
                vix_regime=_vix_regime_full if not _vix_regime_full.empty else None,
                spx_futures_close=indicators.get("spx_futures_close"),
            )
            df["_macro_flag"] = ml_features.macro_override_flag(
                df.index,
                close=_close_full,
                open_=_open_full,
                vix_regime=_vix_regime_full if not _vix_regime_full.empty else None,
                volatility_regime=_vol_regimes_full,
            )
            df["_overnight_w"] = ml_features.compute_overnight_return_weighted(
                _open_full, _close_full
            )

            # ── Broader market context pre-computation ───────────────────────
            # Only when the ablation flag is on.  Computed on the FULL,
            # UNFILTERED df (same principle as the additive features above) so
            # the values for each date are identical to the inference path where
            # the full df is always passed.  The columns survive the
            # meaningful-move filter below and are consumed by build_features()
            # via the `_<name>` column fast-path.
            if settings.LONG_BROADER_CONTEXT_ENABLED:
                _stale_days = int(
                    getattr(settings, "LONG_CONTEXT_STALENESS_MAX_DAYS", 5)
                )
                _vix_level_full = indicators.get(
                    "vix_level", pd.Series(dtype=float)
                )
                _vix_for_term = (
                    _vix_level_full
                    if not _vix_level_full.empty
                    else _close_full
                )
                _ctx_ts, _ctx_tm = ml_features.compute_vix_term_structure(
                    _vix_for_term,
                    vix_short_close=indicators.get("vix_short_close"),
                    vix_long_close=indicators.get("vix_long_close"),
                    staleness_max_days=_stale_days,
                )
                df["_vix_term_slope"] = _ctx_ts.reindex(df.index)
                df["_vix_term_missing"] = _ctx_tm.reindex(df.index)

                _ctx_cs, _ctx_cm = ml_features.compute_credit_stress(
                    df.index,
                    hy_close=indicators.get("credit_hy_close"),
                    ig_close=indicators.get("credit_ig_close"),
                    staleness_max_days=_stale_days,
                )
                df["_credit_stress_score"] = _ctx_cs
                df["_credit_stress_missing"] = _ctx_cm

                _ctx_bs, _ctx_bm = ml_features.compute_market_breadth(
                    df.index,
                    breadth_close=indicators.get("breadth_close"),
                    staleness_max_days=_stale_days,
                )
                df["_breadth_score"] = _ctx_bs
                df["_breadth_missing"] = _ctx_bm

                _ctx_rs, _ctx_rm = ml_features.compute_rates_level(
                    df.index,
                    rates_close=indicators.get("rates_close"),
                    staleness_max_days=_stale_days,
                )
                df["_rates_level_norm"] = _ctx_rs
                df["_rates_missing"] = _ctx_rm

            # ── Target branching ─────────────────────────────────────────────
            # Drawdown-event and three-state targets use different label
            # schemes; they branch here and return early with their own
            # metric dicts so the rest of this method (direction logic) is
            # preserved unchanged.
            _target = settings.LONG_TARGET_TYPE

            if _target == "drawdown_event":
                return self._train_drawdown_event(
                    df, indicators,
                    _ds_total_candles, _ds_date_start, _ds_date_end,
                )
            elif _target == "three_state":
                return self._train_three_state(
                    df, indicators,
                    _ds_total_candles, _ds_date_start, _ds_date_end,
                )
            # else: fall through to the existing direction logic below.

            horizon = int(getattr(settings, "LONG_LABEL_HORIZON_DAYS", 21))
            threshold = float(
                getattr(settings, "LONG_MEANINGFUL_MOVE_THRESHOLD", 0.02)
            )
            df["future_close"] = df["close"].shift(-horizon)
            df.dropna(subset=["future_close"], inplace=True)
            df["forward_return"] = df["future_close"] / df["close"] - 1.0
            labeled_rows = len(df)
            df = df[
                (df["forward_return"] >= threshold)
                | (df["forward_return"] <= -threshold)
            ].copy()
            df["label"] = (df["forward_return"] >= threshold).astype(int)
            excluded_noise_rows = labeled_rows - len(df)
            _ds_labeled_rows = len(df)  # rows that carry a 0/1 label (meaningful moves only)

            # Trim indicators to match df
            trimmed_indicators = {
                k: v.reindex(df.index) if isinstance(v, pd.Series) else v
                for k, v in indicators.items()
            }

            X, weights, valid_pos = self.build_features(df, trimmed_indicators)
            # Align labels by the exact positions that produced feature rows.
            # A plain [:len(X)] slice is wrong when build_features skips rows
            # (e.g. non-positive close) because the skipped positions shift the
            # label assignment for every row that follows them.
            y = df["label"].values[valid_pos]

            min_rows = int(getattr(settings, "LONG_MIN_TRAINING_ROWS", 100))
            if len(X) < min_rows:
                logger.warning("Not enough data to train long-trend model (%d rows)", len(X))
                return {"accuracy": 0.0, "feature_importances": {}}

            # Balance the meaningful-up/down classes so the model cannot
            # achieve a superficially good score by preferring the more common
            # direction in a trending decade.
            class_counts = np.bincount(y.astype(int), minlength=2)
            class_weights = np.ones(2, dtype=np.float32)
            for class_id, count in enumerate(class_counts):
                if count > 0:
                    class_weights[class_id] = len(y) / (2.0 * count)
            weights = weights * class_weights[y.astype(int)]

            # Normalize time-decay and class weights to mean 1.0. With a decade of
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
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=2.0,
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
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        use_label_encoder=False,
                        random_state=42,
                    )

                # Extract VIX regime labels (column index 4 = vix_regime_enc)
                # to enable per-regime OOS breakdown in the walk-forward report.
                try:
                    vix_regime_col = FEATURE_NAMES.index("vix_regime_enc")
                    regime_labels = X[:, vix_regime_col].astype(np.intp)
                except Exception:
                    regime_labels = None

                wf_metrics, oos_probs, oos_labels = (
                    ml_calibration.walk_forward_evaluate(
                        X, y, weights, model_factory=_factory,
                        regime_labels=regime_labels,
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
                ml_calibration.save_calibration_report(
                    calibration_summary,
                    dataset_meta={
                        "total_candles": _ds_total_candles,
                        "labeled_rows": _ds_labeled_rows,
                        "date_start": _ds_date_start,
                        "date_end": _ds_date_end,
                    },
                )
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
            oos_acc = (
                float(calibration_summary["oos_accuracy"])
                if calibration_summary.get("evaluated")
                and calibration_summary.get("oos_accuracy") is not None
                else None
            )
            return {
                # Use purged walk-forward OOS as the headline when available.
                # The chronological holdout is retained separately because it
                # is useful for diagnostics but is not the acceptance metric.
                "accuracy": oos_acc if oos_acc is not None else acc,
                "accuracy_metric": (
                    "purged_walk_forward_oos"
                    if oos_acc is not None
                    else "train"
                ),
                "train_accuracy": acc,
                "target_horizon_days": horizon,
                "meaningful_move_threshold": threshold,
                "training_rows": int(len(X)),
                "excluded_noise_rows": int(excluded_noise_rows),
                "positive_label_rate": float(np.mean(y)),
                "feature_importances": importances,
                "degenerate": degenerate,
                "degeneracy_reason": degeneracy_reason,
                "calibration": calibration_summary,
            }

        except Exception as exc:
            logger.error("Long-trend model training error: %s", exc)
            return {"accuracy": 0.0, "feature_importances": {}}

    # ──────────────────────────────────────────────────────────────────────────
    # Alternative-target training helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def save_promotion_meta(target_type: str) -> None:
        """Persist the target_type of the just-promoted model to the meta sidecar.

        Called by the trainer immediately after a gate-passing model is written
        to MODEL_PATH.  Safe to call from outside the class (e.g. from trainer.py).
        Never raises.
        """
        import json as _json
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(_META_PATH, "w") as _mf:
                _json.dump({"target_type": target_type}, _mf)
            logger.info(
                "ml_model_meta_saved model=long_trend target_type=%s", target_type
            )
        except Exception as exc:
            logger.error("save_promotion_meta error: %s", exc)

    def _train_drawdown_event(
        self,
        df: pd.DataFrame,
        indicators: dict,
        ds_total_candles: int,
        ds_date_start: Optional[str],
        ds_date_end: Optional[str],
    ) -> dict:
        """Train a binary drawdown-event classifier.

        y=1 when the minimum close price in the next LONG_DRAWDOWN_HORIZON
        trading days falls at least LONG_DRAWDOWN_THRESHOLD below today's close.
        Promotion gate: PR-AUC lift ≥ 2× AND precision lift ≥ 2× on purged OOS.
        """
        try:
            import xgboost as xgb

            horizon = int(getattr(settings, "LONG_DRAWDOWN_HORIZON", 21))
            dd_thresh = float(getattr(settings, "LONG_DRAWDOWN_THRESHOLD", 0.05))

            # Build future minimum close and drawdown labels (strictly future)
            future_cols = pd.concat(
                [df["close"].shift(-k) for k in range(1, horizon + 1)], axis=1
            )
            df = df.copy()
            df["_future_min_close"] = future_cols.min(axis=1, skipna=False)
            df.dropna(subset=["_future_min_close"], inplace=True)
            df["_max_drawdown"] = df["_future_min_close"] / df["close"] - 1.0
            df["label"] = (df["_max_drawdown"] <= -dd_thresh).astype(int)
            _ds_labeled_rows = len(df)

            trimmed_indicators = {
                k: v.reindex(df.index) if isinstance(v, pd.Series) else v
                for k, v in indicators.items()
            }
            X, weights, valid_pos = self.build_features(df, trimmed_indicators)
            y = df["label"].values[valid_pos]

            min_rows = int(getattr(settings, "LONG_MIN_TRAINING_ROWS", 100))
            if len(X) < min_rows:
                logger.warning(
                    "Not enough data for drawdown-event model (%d rows)", len(X)
                )
                return {"accuracy": 0.0, "feature_importances": {}}

            # Class-balance: drawdown events are the minority class
            class_counts = np.bincount(y.astype(int), minlength=2)
            class_weights = np.ones(2, dtype=np.float32)
            for cid, cnt in enumerate(class_counts):
                if cnt > 0:
                    class_weights[cid] = len(y) / (2.0 * cnt)
            weights = weights * class_weights[y.astype(int)]
            mean_w = float(weights.mean())
            if mean_w > 0:
                weights = weights / mean_w

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
                X, y, weights, test_size=0.2, shuffle=False
            )
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_lambda=2.0, eval_metric="logloss",
                use_label_encoder=False, random_state=42,
            )
            model.fit(X_train, y_train, sample_weight=w_train,
                      eval_set=[(X_test, y_test)], verbose=False)
            train_acc = float(accuracy_score(y_test, model.predict(X_test)))

            # Walk-forward OOS evaluation (same binary evaluator as direction)
            calibration_summary: dict = {"calibrated": False}
            try:
                def _factory():
                    return xgb.XGBClassifier(
                        n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=2.0, eval_metric="logloss",
                        use_label_encoder=False, random_state=42,
                    )
                wf_metrics, oos_probs, oos_labels = ml_calibration.walk_forward_evaluate(
                    X, y, weights, model_factory=_factory,
                )
                calibration_summary.update(wf_metrics)
                if wf_metrics.get("evaluated"):
                    cal = ml_calibration.fit_calibrator(oos_probs, oos_labels)
                    if cal is not None:
                        calibration_summary["calibrated"] = True
                        calibration_summary["calibration_method"] = cal.method
                        if ml_calibration.save_calibrator(cal):
                            self.calibrator = cal
                            self._calibrator_mtime = None
                ml_calibration.save_calibration_report(
                    calibration_summary,
                    dataset_meta={
                        "total_candles": ds_total_candles,
                        "labeled_rows": _ds_labeled_rows,
                        "date_start": ds_date_start,
                        "date_end": ds_date_end,
                    },
                )
            except Exception as exc:
                logger.error("Drawdown-event calibration error: %s", exc)
                calibration_summary = {"calibrated": False, "reason": str(exc)}

            # Save model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            self.model = model
            self._model_loaded = True

            importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
            degenerate, degeneracy_reason = check_model_degeneracy(model, X)
            if degenerate:
                logger.error(
                    "ml_model_degenerate model=long_trend target=drawdown_event reason=%s",
                    degeneracy_reason,
                )

            event_rate = float(y.mean())
            pr_auc = calibration_summary.get("pr_auc")
            pr_auc_lift = (
                pr_auc / event_rate if pr_auc is not None and event_rate > 0 else None
            )
            logger.info(
                "Drawdown-event model trained: train_acc=%.4f event_rate=%.4f "
                "pr_auc=%s pr_auc_lift=%s",
                train_acc, event_rate, pr_auc, pr_auc_lift,
            )
            oos_acc = (
                float(calibration_summary["oos_accuracy"])
                if calibration_summary.get("evaluated")
                and calibration_summary.get("oos_accuracy") is not None
                else None
            )
            return {
                "accuracy": oos_acc if oos_acc is not None else train_acc,
                "accuracy_metric": (
                    "purged_walk_forward_oos"
                    if oos_acc is not None else "train"
                ),
                "train_accuracy": train_acc,
                "target_type": "drawdown_event",
                "target_horizon_days": horizon,
                "drawdown_threshold": dd_thresh,
                "training_rows": int(len(X)),
                "positive_label_rate": event_rate,
                "pr_auc_lift_vs_prevalence": pr_auc_lift,
                "feature_importances": importances,
                "degenerate": degenerate,
                "degeneracy_reason": degeneracy_reason,
                "calibration": calibration_summary,
            }
        except Exception as exc:
            logger.error("Drawdown-event training error: %s", exc)
            return {"accuracy": 0.0, "feature_importances": {}}

    def _train_three_state(
        self,
        df: pd.DataFrame,
        indicators: dict,
        ds_total_candles: int,
        ds_date_start: Optional[str],
        ds_date_end: Optional[str],
    ) -> dict:
        """Train a three-state (risk-off / neutral / risk-on) classifier.

        Labels: 2=risk-on (fwd_return > threshold), 1=neutral (|return| <= threshold),
        0=risk-off (fwd_return < -threshold).
        Promotion gate: macro-F1 > 0.40 AND each class F1 > 0.25.
        """
        try:
            import xgboost as xgb

            horizon = int(getattr(settings, "LONG_THREE_STATE_HORIZON", 21))
            threshold = float(getattr(settings, "LONG_THREE_STATE_THRESHOLD", 0.02))

            df = df.copy()
            df["_future_close"] = df["close"].shift(-horizon)
            df.dropna(subset=["_future_close"], inplace=True)
            df["_fwd_return"] = df["_future_close"] / df["close"] - 1.0
            conditions = [df["_fwd_return"] > threshold, df["_fwd_return"] < -threshold]
            df["label"] = np.select(conditions, [2, 0], default=1)
            _ds_labeled_rows = len(df)

            trimmed_indicators = {
                k: v.reindex(df.index) if isinstance(v, pd.Series) else v
                for k, v in indicators.items()
            }
            X, weights, valid_pos = self.build_features(df, trimmed_indicators)
            y = df["label"].values[valid_pos].astype(int)

            min_rows = int(getattr(settings, "LONG_MIN_TRAINING_ROWS", 100))
            if len(X) < min_rows:
                logger.warning(
                    "Not enough data for three-state model (%d rows)", len(X)
                )
                return {"accuracy": 0.0, "feature_importances": {}}

            # Three-class balanced weights
            class_counts = np.bincount(y, minlength=3)
            class_weights = np.ones(3, dtype=np.float32)
            for cid, cnt in enumerate(class_counts):
                if cnt > 0:
                    class_weights[cid] = len(y) / (3.0 * cnt)
            weights = weights * class_weights[y]
            mean_w = float(weights.mean())
            if mean_w > 0:
                weights = weights / mean_w

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
                X, y, weights, test_size=0.2, shuffle=False
            )
            model = xgb.XGBClassifier(
                objective="multi:softprob", num_class=3,
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_lambda=2.0, eval_metric="mlogloss",
                use_label_encoder=False, random_state=42,
            )
            model.fit(X_train, y_train, sample_weight=w_train,
                      eval_set=[(X_test, y_test)], verbose=False)
            train_acc = float(accuracy_score(y_test, model.predict(X_test)))

            # Walk-forward multiclass evaluation
            calibration_summary: dict = {"calibrated": False}
            try:
                def _factory():
                    return xgb.XGBClassifier(
                        objective="multi:softprob", num_class=3,
                        n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=2.0, eval_metric="mlogloss",
                        use_label_encoder=False, random_state=42,
                    )
                wf_metrics = _walk_forward_multiclass(
                    X, y, weights, model_factory=_factory,
                    embargo=max(horizon, 21),
                )
                calibration_summary.update(wf_metrics)
                # No probability calibrator for multi-class (collapsed scalar
                # is used directly; isotonic regression is binary-only).
                # Save a calibration report so the staleness auditor has data.
                ml_calibration.save_calibration_report(
                    calibration_summary,
                    dataset_meta={
                        "total_candles": ds_total_candles,
                        "labeled_rows": _ds_labeled_rows,
                        "date_start": ds_date_start,
                        "date_end": ds_date_end,
                    },
                )
            except Exception as exc:
                logger.error("Three-state walk-forward error: %s", exc)
                calibration_summary = {"calibrated": False, "reason": str(exc)}

            # Save model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            self.model = model
            self._model_loaded = True

            importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
            degenerate, degeneracy_reason = check_model_degeneracy(model, X)
            if degenerate:
                logger.error(
                    "ml_model_degenerate model=long_trend target=three_state reason=%s",
                    degeneracy_reason,
                )

            macro_f1 = calibration_summary.get("macro_f1")
            logger.info(
                "Three-state model trained: train_acc=%.4f macro_f1=%s",
                train_acc, macro_f1,
            )
            oos_acc = calibration_summary.get("oos_accuracy")
            return {
                "accuracy": float(oos_acc) if oos_acc is not None else train_acc,
                "accuracy_metric": (
                    "purged_walk_forward_multiclass"
                    if calibration_summary.get("evaluated") else "train"
                ),
                "train_accuracy": train_acc,
                "target_type": "three_state",
                "target_horizon_days": horizon,
                "three_state_threshold": threshold,
                "training_rows": int(len(X)),
                "macro_f1": macro_f1,
                "per_class": calibration_summary.get("per_class"),
                "feature_importances": importances,
                "degenerate": degenerate,
                "degeneracy_reason": degeneracy_reason,
                "calibration": calibration_summary,
            }
        except Exception as exc:
            logger.error("Three-state training error: %s", exc)
            return {"accuracy": 0.0, "feature_importances": {}}

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> float:
        """
        Return a directional confidence score in [0, 1] where > 0.5 is bullish.

        Interpretation by target_type:
          direction      — P(BUY) exactly as before.
          drawdown_event — 1 − calibrated_P(drawdown): calibrate raw P(drawdown)
                           first (the calibrator is fitted on those raw values),
                           then invert so a high score = no drawdown expected.
          three_state    — P(risk-on) + 0.5 × P(neutral): risk-on pushes the
                           score above 0.5; risk-off pushes it below. No
                           probability calibrator (binary-only fitting).

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
            # Keep the APK-facing prediction path alive while an older
            # 15-feature pickle is being replaced. New models use all 19
            # features; legacy models consume the original stable prefix.
            expected = self._model_feature_count
            if expected is not None and features.shape[1] != expected:
                if expected < features.shape[1] and expected == 15:
                    features = features[:, :expected]
                else:
                    raise ValueError(
                        f"model expects {expected} features, got {features.shape[1]}"
                    )

            raw_proba = self.model.predict_proba(features)[0]
            target = self.target_type

            if target == "three_state":
                # raw_proba: [P(0=risk-off), P(1=neutral), P(2=risk-on)]
                # Collapse to a single [0,1] score where > 0.5 = bullish.
                # No calibrator: isotonic regression is binary-only.
                prob = float(raw_proba[2]) + 0.5 * float(raw_proba[1])

            elif target == "drawdown_event":
                # Calibrator was fitted on raw P(drawdown) with label=1 meaning
                # drawdown.  Apply it to the RAW value first so calibration is
                # consistent with training, then invert so > 0.5 = bullish.
                raw_dd = float(raw_proba[1])
                if self.calibrator is not None:
                    try:
                        raw_dd = self.calibrator.transform(raw_dd)
                    except Exception as exc:
                        logger.error("Long-trend drawdown calibration apply error: %s", exc)
                prob = 1.0 - raw_dd

            else:
                # direction (binary): P(BUY = class 1)
                prob = float(raw_proba[1])
                # Apply the persisted probability calibrator when available.
                # Raw probability is the fallback — calibration must never
                # break prediction.
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

    def predict_class_probs(self, features: np.ndarray) -> Optional[list]:
        """Return raw class probabilities for three_state models, None otherwise.

        For direction and drawdown_event models, callers should use predict()
        which returns an already-collapsed directional confidence.

        Args:
            features: 1D or 2D ndarray of shape (n_features,) or (1, n_features)

        Returns:
            List of floats [P(risk-off), P(neutral), P(risk-on)] for three_state,
            or None for other target types or when the model is unavailable.
        """
        try:
            self._maybe_reload()
            if self.model is None or self.target_type != "three_state":
                return None
            if features.ndim == 1:
                features = features.reshape(1, -1)
            raw = self.model.predict_proba(features)[0]
            return [round(float(p), 4) for p in raw]
        except Exception as exc:
            logger.error("predict_class_probs error: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Model persistence
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        """
        Load model from ml/models/long_trend_model.pkl if it exists.

        Returns:
            True if loaded successfully, False otherwise.

        Baseline-mode logic:
          - No pkl file → _baseline_mode = True, model = None.
          - Legacy 15-feature pkl (OOS lift ≈ −29 pp) → treated as no trained
            edge; _baseline_mode = True, model = None. The file is left on disk
            so a future gate-passing retrain can overwrite it.
          - Wrong feature count (not 15, not 19) → model = None (existing
            behaviour); _baseline_mode = True for honesty.
          - Load error → _baseline_mode = True.
          - 19-feature pkl loaded, but last recorded successful training used a
            pre-gate metric ("train" instead of "purged_walk_forward_oos") →
            _baseline_mode = True, model = None.  This catches the rollback
            artifact: a model that was trained before the OOS gate existed and
            has never cleared the lift check.  Clears automatically when a
            future gate-passing retrain records accuracy_metric=
            "purged_walk_forward_oos" and overwrites the pkl.
          - 19-feature pkl loaded and last successful metric is
            "purged_walk_forward_oos" → _baseline_mode = False.
          - Meta sidecar target_type differs from settings.LONG_TARGET_TYPE →
            _baseline_mode = True, model = None.  Forces a retrain with the
            new target before the gauge switches behaviour.
        """
        try:
            if MODEL_PATH.exists():
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
                n_in = getattr(model, "n_features_in_", None)
                self._model_feature_count = int(n_in) if n_in is not None else len(FEATURE_NAMES)
                is_legacy_xgb = (
                    n_in is not None
                    and int(n_in) == 15
                    and type(model).__module__.split(".")[0] == "xgboost"
                )
                if is_legacy_xgb:
                    # Legacy model has −29 pp OOS lift; do not serve it.
                    # Leave the pkl on disk — a promoted model will overwrite it.
                    logger.warning(
                        "ml_model_baseline_mode model=long_trend expected_features=%d "
                        "found=%d reason=legacy_model_no_oos_edge "
                        "action=calibrated_base_rate",
                        len(FEATURE_NAMES), int(n_in),
                    )
                    self.model = None
                    self._model_feature_count = None
                    self._baseline_mode = True
                    self._model_loaded = True
                    return False
                if n_in is not None and int(n_in) != len(FEATURE_NAMES):
                    logger.warning(
                        "ml_model_stale model=long_trend expected_features=%d "
                        "found=%d action=baseline_await_retrain",
                        len(FEATURE_NAMES), int(n_in),
                    )
                    self.model = None
                    self._model_feature_count = None
                    self._baseline_mode = True
                    self._model_loaded = True
                    return False
                # Gate-pass check: verify the pkl was trained by a run that
                # cleared the OOS lift gate.  The training status carries
                # `last_success_accuracy_metric`; anything other than
                # "purged_walk_forward_oos" means the "last good" model predates
                # the gate and has never been validated against it.
                last_metric = get_last_successful_accuracy_metric("long_trend")
                if last_metric not in ("purged_walk_forward_oos", "purged_walk_forward_multiclass"):
                    logger.warning(
                        "ml_model_baseline_mode model=long_trend "
                        "last_success_metric=%s "
                        "reason=pre_gate_artifact_never_cleared_oos_lift "
                        "action=calibrated_base_rate",
                        last_metric,
                    )
                    self.model = None
                    self._model_feature_count = None
                    self._baseline_mode = True
                    self._model_loaded = True
                    return False

                # Target-type alignment: read the meta sidecar written by the
                # last successful promotion and compare to the configured target.
                # A mismatch forces baseline mode until a retrain for the new
                # target completes — this prevents a direction model from being
                # served as a drawdown/three_state model (wrong output semantics).
                try:
                    import json as _json
                    if _META_PATH.exists():
                        with open(_META_PATH) as _mf:
                            _meta = _json.load(_mf)
                        _promoted = str(_meta.get("target_type", "direction"))
                    else:
                        _promoted = "direction"  # pre-meta pkl defaults to direction
                    self._promoted_target_type = _promoted
                    configured = settings.LONG_TARGET_TYPE
                    if _promoted != configured:
                        logger.warning(
                            "ml_model_baseline_mode model=long_trend "
                            "promoted_target=%s configured_target=%s "
                            "reason=target_type_mismatch action=baseline_await_retrain",
                            _promoted, configured,
                        )
                        self.model = None
                        self._model_feature_count = None
                        self._baseline_mode = True
                        self._model_loaded = True
                        return False
                except Exception as _meta_exc:
                    logger.warning(
                        "ml_model_meta_read_error model=long_trend error=%s "
                        "— assuming direction target",
                        _meta_exc,
                    )
                    self._promoted_target_type = "direction"

                self.model = model
                self._baseline_mode = False
                self._model_loaded = True
                logger.info(
                    "Long-trend model loaded from %s target_type=%s",
                    MODEL_PATH,
                    self.target_type,
                )
                return True
            else:
                logger.info(
                    "No long-trend model file found at %s — baseline mode active",
                    MODEL_PATH,
                )
                self._baseline_mode = True
                self._model_loaded = True  # Prevent repeated load attempts
                return False
        except Exception as exc:
            logger.error("Error loading long-trend model: %s", exc)
            self._baseline_mode = True
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
            X, _, _ = self.build_features(df, indicators)
            if len(X) == 0:
                return None
            return X[-1:].astype(np.float32)
        except Exception as exc:
            logger.error("build_latest_features error: %s", exc)
            return None
