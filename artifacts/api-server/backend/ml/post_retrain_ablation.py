"""
Post-retrain broader-context ablation runner.

Called automatically after a successful long-trend retrain to keep the
19-feature vs 27-feature comparison current.  Results are *appended*
(not overwritten) to ml/models/ablation_broader_context.json with a UTC
timestamp so the full history of retrain verdicts is preserved.

Design constraints
------------------
* No module-level side effects and no global state mutation.  Unlike the
  standalone scripts/ablate_broader_context.py (which patches calibration
  paths and settings at import time for safety as a CLI script), this module
  is imported into the live server process and must not mutate shared config.
* The 19-feature baseline matrix is obtained by slicing the first
  len(_BASE_FEATURE_NAMES) columns from whatever LongTrendModel.build_features()
  returns.  build_features() always emits base features first; the 8 context
  columns follow only when LONG_BROADER_CONTEXT_ENABLED=True.  This slice
  approach is safe under concurrent inference because it never writes to
  shared settings.
* Never raises — all exceptions are caught and logged so a failed ablation
  does not abort or delay the retrain that triggered it.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_ABLATION_JSON = _MODELS_DIR / "ablation_broader_context.json"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_factory():
    """Return a fresh XGBClassifier factory (callable → estimator)."""
    def factory():
        import xgboost as xgb  # noqa: PLC0415
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
    return factory


def _build_matrices(
    daily_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    spx_close: pd.Series,
    broader_context: dict,
    horizon: int,
    threshold: float,
):
    """
    Build X_19 (19-feature baseline) and X_27 (+ 8 broader-context features)
    from the supplied data frames.

    Returns:
        X_19, X_27, y, weights, timestamps, majority_baseline, base_names, ctx_names

    No global state is mutated.  build_features() is called with the live
    settings.LONG_BROADER_CONTEXT_ENABLED flag intact.  Because
    build_features() always emits the 19 base features first (followed by the
    optional 8 context features when the flag is True), the 19-feature
    baseline is obtained by slicing X[:, :len(_BASE_FEATURE_NAMES)] regardless
    of the current flag value.  The 8 context columns are then appended
    explicitly from the pre-computed context series to form X_27.

    Raises ValueError when fewer than 80 labeled rows are available.
    """
    from config import settings  # noqa: PLC0415 — lazy import to defer flag read
    from indicators.technical import TechnicalIndicators  # noqa: PLC0415
    import ml.features as ml_features  # noqa: PLC0415
    from ml.long_trend import (  # noqa: PLC0415
        _BASE_FEATURE_NAMES,
        _BROADER_CONTEXT_FEATURE_NAMES,
        LongTrendModel,
    )

    # ── Compute technical indicators ──────────────────────────────────────────
    ti = TechnicalIndicators()
    indicators = ti.compute_all(daily_df, vix_df, exclude_extended=True)
    if not spx_close.empty:
        indicators["spx_futures_close"] = spx_close
    # Inject context series so build_features() can see them; it only reads
    # them when the flag is True, but we need them in `indicators` for the
    # manual X_27 build below.
    for k, v in broader_context.items():
        indicators[k] = v

    # ── Pre-compute temporal features on the FULL unfiltered frame ───────────
    df = daily_df.copy()
    if "is_extended_hours" in df.columns:
        df = df[df["is_extended_hours"] == False].copy()  # noqa: E712

    df["_return_5d"]  = df["close"].pct_change(5)
    df["_return_10d"] = df["close"].pct_change(10)
    df["_return_20d"] = df["close"].pct_change(20)
    df["_vol_avg20"]  = df["volume"].rolling(20).mean() if "volume" in df.columns else 0.0

    _close = df["close"]
    _open  = df["open"] if "open" in df.columns else _close
    _liq   = df["liquidity_class"] if "liquidity_class" in df.columns else None
    _atr   = indicators.get("atr", pd.Series(dtype=float))
    _vix_r = indicators.get("vix_regime", pd.Series(dtype=object))

    _vol_reg = ml_features.compute_volatility_regime(_close, atr=_atr, liquidity_class=_liq)
    df["_vol_regime_enc"] = ml_features.encode_volatility_regime(_vol_reg)
    df["_macro_sens"]     = ml_features.compute_macro_sensitivity(
        _close, open_=_open,
        vix_regime=_vix_r if not _vix_r.empty else None,
        spx_futures_close=indicators.get("spx_futures_close"),
    )
    df["_macro_flag"]     = ml_features.macro_override_flag(
        df.index, close=_close, open_=_open,
        vix_regime=_vix_r if not _vix_r.empty else None,
        volatility_regime=_vol_reg,
    )
    df["_overnight_w"]    = ml_features.compute_overnight_return_weighted(_open, _close)

    # ── Broader-context features on the FULL frame ───────────────────────────
    _stale_days = int(getattr(settings, "LONG_CONTEXT_STALENESS_MAX_DAYS", 5))
    _vix_level  = indicators.get("vix_level", pd.Series(dtype=float))
    _vix_proxy  = _vix_level if not _vix_level.empty else _close

    _ctx_ts, _ctx_tm = ml_features.compute_vix_term_structure(
        _vix_proxy,
        vix_short_close=broader_context.get("vix_short_close"),
        vix_long_close=broader_context.get("vix_long_close"),
        staleness_max_days=_stale_days,
    )
    _ctx_cs, _ctx_cm = ml_features.compute_credit_stress(
        df.index,
        hy_close=broader_context.get("credit_hy_close"),
        ig_close=broader_context.get("credit_ig_close"),
        staleness_max_days=_stale_days,
    )
    _ctx_bs, _ctx_bm = ml_features.compute_market_breadth(
        df.index,
        breadth_close=broader_context.get("breadth_close"),
        staleness_max_days=_stale_days,
    )
    _ctx_rs, _ctx_rm = ml_features.compute_rates_level(
        df.index,
        rates_close=broader_context.get("rates_close"),
        staleness_max_days=_stale_days,
    )

    # Store context columns so they survive the label filter below
    df["_vix_term_slope"]        = _ctx_ts.reindex(df.index)
    df["_vix_term_missing"]      = _ctx_tm.reindex(df.index)
    df["_credit_stress_score"]   = _ctx_cs.reindex(df.index)
    df["_credit_stress_missing"] = _ctx_cm.reindex(df.index)
    df["_breadth_score"]         = _ctx_bs.reindex(df.index)
    df["_breadth_missing"]       = _ctx_bm.reindex(df.index)
    df["_rates_level_norm"]      = _ctx_rs.reindex(df.index)
    df["_rates_missing"]         = _ctx_rm.reindex(df.index)

    # ── Labels: direction over `horizon` days, exclude near-flat ─────────────
    df["_future_close"] = df["close"].shift(-horizon)
    df = df.dropna(subset=["_future_close"]).copy()
    df["_fwd_ret"] = df["_future_close"] / df["close"] - 1.0
    df = df[(df["_fwd_ret"] >= threshold) | (df["_fwd_ret"] <= -threshold)].copy()
    df["_label"]   = (df["_fwd_ret"] >= threshold).astype(int)

    if len(df) < 80:
        raise ValueError(f"Too few labeled rows ({len(df)}) — need ≥ 80.")

    # ── Build feature matrix via LongTrendModel.build_features ───────────────
    # build_features() always lays out base features first, followed by the
    # optional 8 context features when LONG_BROADER_CONTEXT_ENABLED=True.
    # Slicing [:, :n_base] gives the 19-feature baseline without touching any
    # global settings — no race condition, no flag mutation.
    n_base = len(_BASE_FEATURE_NAMES)  # 19

    trimmed_ind = {
        k: v.reindex(df.index) if isinstance(v, pd.Series) else v
        for k, v in indicators.items()
    }
    model = LongTrendModel()
    X_full, weights_raw, valid_pos = model.build_features(df, trimmed_ind)
    # Extract the 19 base features from whatever build_features returned.
    # When the flag is False:  X_full.shape == (n, 19) → slice is a no-op.
    # When the flag is True:   X_full.shape == (n, 27) → slice drops context cols.
    X_19 = X_full[:, :n_base]

    y = df["_label"].values[valid_pos]

    # ── Append the 8 context columns → X_27 ──────────────────────────────────
    ctx_cols = [
        "_vix_term_slope", "_vix_term_missing",
        "_credit_stress_score", "_credit_stress_missing",
        "_breadth_score", "_breadth_missing",
        "_rates_level_norm", "_rates_missing",
    ]
    ctx_rows = []
    for pos in valid_pos:
        row = []
        for col in ctx_cols:
            try:
                val = float(df[col].iloc[pos])
                row.append(
                    val if math.isfinite(val)
                    else (0.5 if ("score" in col or "slope" in col or "norm" in col) else 1.0)
                )
            except Exception:
                row.append(1.0)
        ctx_rows.append(row)
    X_ctx = np.array(ctx_rows, dtype=np.float32)
    X_27  = np.hstack([X_19, X_ctx])

    # ── Timestamps for metadata ───────────────────────────────────────────────
    timestamps = df.index[valid_pos]

    # ── Normalize sample weights ──────────────────────────────────────────────
    class_counts = np.bincount(y.astype(int), minlength=2)
    class_w = np.ones(2, dtype=np.float32)
    for cid, cnt in enumerate(class_counts):
        if cnt > 0:
            class_w[cid] = len(y) / (2.0 * cnt)
    weights = weights_raw * class_w[y.astype(int)]
    mw = float(weights.mean())
    if mw > 0:
        weights = weights / mw

    positive_rate    = float(y.mean())
    majority_baseline = max(positive_rate, 1.0 - positive_rate)

    return X_19, X_27, y, weights, timestamps, majority_baseline, _BASE_FEATURE_NAMES, _BROADER_CONTEXT_FEATURE_NAMES


def _append_to_json(result: dict, out_path: Path, max_history: int = 52) -> None:
    """
    Append *result* to out_path as a JSON list entry.

    If the file already exists and contains a JSON object (legacy single-run
    format from the CLI script), it is promoted to a one-element list before
    appending.  If the file is absent, a new single-element list is written.

    After appending, only the *max_history* most-recent entries are kept so
    the file does not grow without bound across weekly retrains.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if out_path.exists():
        try:
            with open(out_path) as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                existing = raw
            elif isinstance(raw, dict):
                # Promote legacy single-report to list
                existing = [raw]
        except Exception as exc:
            logger.warning(
                "ablation_append: could not read %s (%s) — starting fresh list",
                out_path, exc,
            )
    existing.append(result)
    if max_history > 0 and len(existing) > max_history:
        existing = existing[-max_history:]
    with open(out_path, "w") as fh:
        json.dump(existing, fh, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_broader_context_ablation(
    daily_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    spx_close: pd.Series,
    broader_context: dict,
    *,
    horizon: int = 21,
    threshold: float = 0.02,
    n_splits: int = 5,
    out_path: Optional[Path] = None,
    max_history: int = 52,
) -> dict:
    """
    Run the 19- vs 27-feature broader-context ablation on data already
    loaded for a retrain and append a timestamped record to the audit JSON.

    Args:
        daily_df:        Daily VOO candles with DatetimeIndex (same frame
                         used for training).
        vix_df:          Daily VIX candles (may be empty).
        spx_close:       SPX futures daily close series (may be empty).
        broader_context: Dict mapping indicator keys to daily close Series.
                         Keys: vix_short_close, vix_long_close,
                         credit_hy_close, credit_ig_close, breadth_close,
                         rates_close.  Empty Series triggers the missing
                         fallback (same as training time).
        horizon:         Label horizon in trading days (default 21).
        threshold:       Meaningful-move threshold (default 0.02 = 2%).
        n_splits:        Walk-forward fold count (default 5).
        out_path:        Override the output JSON path.
        max_history:     Maximum number of entries kept in the audit JSON
                         (default 52 — roughly one year of weekly retrains).
                         Oldest entries are trimmed when the cap is exceeded.

    Returns:
        The new ablation result dict that was appended to the JSON file.
        Returns {} on hard failure so callers can check truthiness.

    Never raises.
    """
    from config import settings  # noqa: PLC0415
    from ml.calibration import walk_forward_evaluate  # noqa: PLC0415

    out_path = Path(out_path) if out_path else _ABLATION_JSON
    run_ts   = datetime.now(tz=timezone.utc).isoformat()
    oos_gate = float(getattr(settings, "LONG_MIN_OOS_ACCURACY_LIFT", 0.0))

    logger.info(
        "ablation_start horizon=%d threshold=%.2f n_splits=%d oos_gate=%+.4f",
        horizon, threshold, n_splits, oos_gate,
    )

    # ── Feature matrices ──────────────────────────────────────────────────────
    try:
        (
            X_19, X_27, y, weights, timestamps,
            majority_baseline,
            base_names, ctx_names,
        ) = _build_matrices(
            daily_df, vix_df, spx_close, broader_context,
            horizon=horizon, threshold=threshold,
        )
    except Exception as exc:
        logger.error("ablation_build_matrices_failed error=%s", exc)
        return {}

    n_rows        = len(y)
    positive_rate = float(y.mean())
    logger.info(
        "ablation_matrices n_labeled=%d positive_rate=%.3f majority_baseline=%.4f "
        "date_range=%s … %s",
        n_rows, positive_rate, majority_baseline,
        timestamps[0].date(), timestamps[-1].date(),
    )

    # ── Walk-forward OOS evaluation — identical folds for both arms ───────────
    embargo = max(horizon, 21)
    try:
        logger.info("ablation_wf_start arm=19feat n_splits=%d embargo=%d", n_splits, embargo)
        metrics_19, _, _ = walk_forward_evaluate(
            X_19, y, weights, model_factory=_xgb_factory(),
            n_splits=n_splits, embargo=embargo,
        )
    except Exception as exc:
        logger.error("ablation_wf_failed arm=19feat error=%s", exc)
        return {}

    try:
        logger.info("ablation_wf_start arm=27feat n_splits=%d embargo=%d", n_splits, embargo)
        metrics_27, _, _ = walk_forward_evaluate(
            X_27, y, weights, model_factory=_xgb_factory(),
            n_splits=n_splits, embargo=embargo,
        )
    except Exception as exc:
        logger.error("ablation_wf_failed arm=27feat error=%s", exc)
        return {}

    if not metrics_19.get("evaluated"):
        logger.error("ablation_wf_not_evaluated arm=19feat reason=%s", metrics_19.get("reason"))
        return {}
    if not metrics_27.get("evaluated"):
        logger.error("ablation_wf_not_evaluated arm=27feat reason=%s", metrics_27.get("reason"))
        return {}

    # ── Key metrics ───────────────────────────────────────────────────────────
    acc_19  = float(metrics_19["oos_accuracy"])
    acc_27  = float(metrics_27["oos_accuracy"])
    bal_19  = metrics_19.get("oos_balanced_accuracy")
    bal_27  = metrics_27.get("oos_balanced_accuracy")
    lift_19 = acc_19 - majority_baseline
    lift_27 = acc_27 - majority_baseline
    delta   = acc_27 - acc_19
    bal_delta = (
        float(bal_27) - float(bal_19)
        if bal_19 is not None and bal_27 is not None
        else None
    )

    # ── Gate check ────────────────────────────────────────────────────────────
    passes_gate = (lift_27 >= oos_gate) and (delta > 0.0)

    # ── Structured log — always visible in health dashboard ───────────────────
    if passes_gate:
        logger.warning(
            "ablation_broader_context_gate_pass "
            "acc_19=%.4f acc_27=%.4f delta=%+.4f lift_27=%+.4f oos_gate=%+.4f "
            "recommendation=Enable_LONG_BROADER_CONTEXT_ENABLED",
            acc_19, acc_27, delta, lift_27, oos_gate,
        )
    else:
        reasons = []
        if delta <= 0.0:
            reasons.append(f"delta={delta:+.4f} (27-feat does not beat 19-feat baseline)")
        if lift_27 < oos_gate:
            reasons.append(f"lift_27={lift_27:+.4f} < gate={oos_gate:+.4f}")
        logger.info(
            "ablation_broader_context_gate_fail "
            "acc_19=%.4f acc_27=%.4f delta=%+.4f lift_27=%+.4f oos_gate=%+.4f "
            "reasons=%s",
            acc_19, acc_27, delta, lift_27, oos_gate,
            "; ".join(reasons),
        )

    # ── Context feature importances (full-data diagnostic) ───────────────────
    context_feature_importances: dict = {}
    try:
        import xgboost as xgb  # noqa: PLC0415
        full_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=2.0, eval_metric="logloss",
            use_label_encoder=False, random_state=42,
        )
        full_model.fit(X_27, y, sample_weight=weights)
        all_names = base_names + ctx_names
        all_imps  = full_model.feature_importances_.tolist()
        context_feature_importances = {
            name: round(float(imp), 6)
            for name, imp in zip(all_names, all_imps)
            if name in ctx_names
        }
    except Exception as exc:
        logger.warning("ablation_importances_error error=%s", exc)

    # ── Build result record ───────────────────────────────────────────────────
    result: dict = {
        "ablation": "broader_context_features",
        "run_timestamp_utc": run_ts,
        "trigger": "post_retrain",
        "data_source": "db",
        "horizon_days": horizon,
        "meaningful_move_threshold": threshold,
        "n_splits": n_splits,
        "embargo_rows": embargo,
        "n_labeled_rows": n_rows,
        "date_range_start": str(timestamps[0].date()),
        "date_range_end":   str(timestamps[-1].date()),
        "positive_rate": round(positive_rate, 4),
        "majority_baseline": round(majority_baseline, 4),
        "LONG_MIN_OOS_ACCURACY_LIFT": oos_gate,
        "baseline_19feat": {
            "n_features": int(X_19.shape[1]),
            "oos_accuracy": round(acc_19, 4),
            "oos_lift_vs_majority": round(lift_19, 4),
            "oos_balanced_accuracy": round(float(bal_19), 4) if bal_19 is not None else None,
            "folds": metrics_19.get("folds"),
        },
        "candidate_27feat": {
            "n_features": int(X_27.shape[1]),
            "oos_accuracy": round(acc_27, 4),
            "oos_lift_vs_majority": round(lift_27, 4),
            "oos_balanced_accuracy": round(float(bal_27), 4) if bal_27 is not None else None,
            "folds": metrics_27.get("folds"),
            "context_feature_importances": context_feature_importances,
        },
        "accuracy_delta_27_minus_19": round(delta, 4),
        "balanced_accuracy_delta": round(bal_delta, 4) if bal_delta is not None else None,
        "passes_promotion_gate": passes_gate,
        "promotion_gate_description": (
            "27-feat OOS lift >= LONG_MIN_OOS_ACCURACY_LIFT "
            "AND 27-feat OOS accuracy > 19-feat OOS accuracy"
        ),
        "recommendation": (
            "Enable LONG_BROADER_CONTEXT_ENABLED=True after a gate-passing retrain."
            if passes_gate
            else "Keep LONG_BROADER_CONTEXT_ENABLED=False — broader context does not improve OOS accuracy."
        ),
    }

    # ── Append to audit JSON ──────────────────────────────────────────────────
    try:
        _append_to_json(result, out_path, max_history=max_history)
        logger.info("ablation_report_appended path=%s passes_gate=%s", out_path, passes_gate)
    except Exception as exc:
        logger.error("ablation_report_write_failed path=%s error=%s", out_path, exc)

    # ── Promotion signal + optional auto-enable ───────────────────────────────
    # When the gate passes for the first time, persist a promotion record so
    # the healthz endpoint and operators know a gate-passing retrain occurred.
    # If LONG_BROADER_CONTEXT_AUTO_ENABLE=True, flip the in-memory flag
    # immediately so the next scheduled retrain trains the 27-feature model
    # without manual intervention.
    if passes_gate:
        try:
            from ml.training_status import record_broader_context_promotion  # noqa: PLC0415
            from config import settings as _cfg  # noqa: PLC0415

            auto_enabled = False
            if getattr(_cfg, "LONG_BROADER_CONTEXT_AUTO_ENABLE", False):
                # Only flip if not already enabled; idempotent re-enables are fine
                # but we log clearly to avoid confusion.
                if not _cfg.LONG_BROADER_CONTEXT_ENABLED:
                    _cfg.LONG_BROADER_CONTEXT_ENABLED = True  # type: ignore[assignment]
                    auto_enabled = True
                    logger.warning(
                        "ablation_broader_context_auto_enabled: "
                        "LONG_BROADER_CONTEXT_ENABLED flipped True in-memory "
                        "(LONG_BROADER_CONTEXT_AUTO_ENABLE=True). "
                        "Next retrain will use the 27-feature model. "
                        "Set LONG_BROADER_CONTEXT_ENABLED=True in the environment "
                        "for this to persist across server restarts."
                    )
                else:
                    auto_enabled = True
                    logger.info(
                        "ablation_broader_context_auto_enabled: flag already True"
                    )

            record_broader_context_promotion(
                delta=delta,
                lift=lift_27,
                acc_27=acc_27,
                auto_enabled=auto_enabled,
            )
        except Exception as _promo_exc:
            logger.error("ablation_promotion_record_failed error=%s", _promo_exc)

    return result
