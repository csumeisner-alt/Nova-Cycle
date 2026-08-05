"""
Broader-Context Feature Ablation
=================================
Compares OOS accuracy of the 19-feature baseline vs the 27-feature model
(19 base + 8 credit/rates/breadth/term-structure features) over the same
historical period and the same purged walk-forward folds.

The flag LONG_BROADER_CONTEXT_ENABLED is only recommended for production if
the 27-feature model clears the LONG_MIN_OOS_ACCURACY_LIFT gate AND beats
the 19-feature baseline on OOS accuracy.

Usage (run from artifacts/api-server/backend/):

    python scripts/ablate_broader_context.py          # use DB (novacycle.db)
    python scripts/ablate_broader_context.py --yf     # fetch from yfinance
    python scripts/ablate_broader_context.py --db PATH --out PATH

Output:
    ml/models/ablation_broader_context.json   — auditable comparison report
    Console table + PASS / FAIL recommendation

NEVER writes to ml/models/*.pkl or touches the live model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Patch calibration MODEL_DIR BEFORE any ml imports so no pkl/json is ever
# written to ml/models/ during this script's run.
import ml.calibration as _cal_mod

_DRY_DIR = Path(tempfile.mkdtemp(prefix="ablation_ctx_"))
_cal_mod.MODEL_DIR = _DRY_DIR
_cal_mod.CALIBRATOR_PATH = _DRY_DIR / "long_trend_calibrator.pkl"
_cal_mod.REPORT_PATH = _DRY_DIR / "long_trend_calibration.json"

# Force the 19-feature FEATURE_NAMES BEFORE importing long_trend so the
# module-level list is computed with the ablation flag off.
from config import settings as _settings

_settings.LONG_BROADER_CONTEXT_ENABLED = False  # type: ignore[assignment]

from ml.calibration import walk_forward_evaluate
from ml.long_trend import (
    _BASE_FEATURE_NAMES,
    _BROADER_CONTEXT_FEATURE_NAMES,
    FEATURE_NAMES,
    VIX_REGIME_MAP,
    LongTrendModel,
)
import ml.features as ml_features
from indicators.technical import TechnicalIndicators

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ablate_ctx")

# ── Audit output path (real ml/models dir — JSON only, no pkl) ────────────────
_REAL_MODELS_DIR = BACKEND / "ml" / "models"
_DEFAULT_OUT = _REAL_MODELS_DIR / "ablation_broader_context.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_db(db_path: str):
    """Read VOO, VIX, SPX from the SQLite DB (read-only)."""
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)

    voo = pd.read_sql_query(
        """
        SELECT timestamp, open, high, low, close, volume,
               is_extended_hours, session_type
        FROM voo_candles
        WHERE ticker='VOO' AND timeframe='daily' AND is_extended_hours=0
        ORDER BY timestamp ASC
        """,
        con, parse_dates=["timestamp"], index_col="timestamp",
    )
    vix = pd.read_sql_query(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM vix_candles
        WHERE ticker='^VIX' AND timeframe='daily'
        ORDER BY timestamp ASC
        """,
        con, parse_dates=["timestamp"], index_col="timestamp",
    )
    spx_rows = pd.read_sql_query(
        """
        SELECT timestamp, close FROM spx_candles
        WHERE timeframe='daily' ORDER BY timestamp ASC
        """,
        con, parse_dates=["timestamp"], index_col="timestamp",
    )
    con.close()

    for df in (voo, vix):
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")]

    voo.index = pd.to_datetime(voo.index)
    voo = voo[~voo.index.duplicated(keep="last")]
    vix.index = pd.to_datetime(vix.index)
    vix = vix[~vix.index.duplicated(keep="last")]

    spx = pd.Series(dtype=float)
    if not spx_rows.empty:
        spx_rows.index = pd.to_datetime(spx_rows.index)
        spx = spx_rows["close"]
        spx = spx[~spx.index.duplicated(keep="last")]

    return voo, vix, spx


def _load_from_yfinance(start: str = "2015-01-01"):
    """Fetch VOO, VIX, SPX + context tickers from yfinance."""
    import yfinance as yf  # noqa: PLC0415

    def _fetch(ticker, cols=None):
        df = yf.download(ticker, start=start, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        return df[cols] if cols else df

    voo = _fetch("VOO", ["open", "high", "low", "close", "volume"])
    vix = _fetch("^VIX", ["open", "high", "low", "close"])
    vix["volume"] = 0.0
    spx = _fetch("ES=F", ["close"])["close"].dropna()

    # Broader-context tickers (used for the 27-feature arm only)
    ctx: dict[str, pd.Series] = {}
    ctx_tickers = {
        "vix_short_close":  _settings.VIX_SHORT_TICKER,   # ^VIX9D
        "vix_long_close":   _settings.VIX_LONG_TICKER,    # ^VIX3M
        "credit_hy_close":  _settings.CREDIT_HY_TICKER,   # HYG
        "credit_ig_close":  _settings.CREDIT_IG_TICKER,   # LQD
        "breadth_close":    _settings.BREADTH_TICKER,      # ^NYAD
        "rates_close":      _settings.RATES_TICKER,        # ^TNX
    }
    for key, ticker in ctx_tickers.items():
        if not ticker:
            continue
        try:
            raw = _fetch(ticker, None)
            if "close" in raw.columns:
                ctx[key] = raw["close"].dropna()
        except Exception as exc:
            logger.warning("Could not fetch %s (%s): %s", key, ticker, exc)

    return voo, vix, spx, ctx


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_matrices(
    voo: pd.DataFrame,
    vix: pd.DataFrame,
    spx: pd.Series,
    ctx: dict[str, pd.Series],
    horizon: int = 21,
    threshold: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build X_19 (base) and X_27 (base + context) feature matrices plus
    aligned labels and weights over the same OOS-eligible rows.

    Returns:
        X_19        [n, 19]
        X_27        [n, 27]
        y           [n]   binary direction labels
        weights     [n]   time-decay sample weights
        timestamps  [n]   pd.Timestamp for each row
        majority_baseline  float  max(positive_rate, 1-positive_rate)
    """
    # ── Indicators ────────────────────────────────────────────────────────────
    ti = TechnicalIndicators()
    indicators = ti.compute_all(voo, vix, exclude_extended=True)
    if not spx.empty:
        indicators["spx_futures_close"] = spx
    # Inject context indicators so build_features() can see them (even though
    # LONG_BROADER_CONTEXT_ENABLED=False, we store them for X_27 below).
    for k, v in ctx.items():
        indicators[k] = v

    # ── Pre-compute temporal features on the FULL unfiltered frame ────────────
    df = voo.copy()
    if "is_extended_hours" in df.columns:
        df = df[df["is_extended_hours"] == False].copy()

    df["_return_5d"]  = df["close"].pct_change(5)
    df["_return_10d"] = df["close"].pct_change(10)
    df["_return_20d"] = df["close"].pct_change(20)
    df["_vol_avg20"]  = df["volume"].rolling(20).mean() if "volume" in df.columns else 0.0

    _close  = df["close"]
    _open   = df["open"] if "open" in df.columns else _close
    _liq    = df["liquidity_class"] if "liquidity_class" in df.columns else None
    _atr    = indicators.get("atr", pd.Series(dtype=float))
    _vix_r  = indicators.get("vix_regime", pd.Series(dtype=object))

    _vol_reg = ml_features.compute_volatility_regime(_close, atr=_atr, liquidity_class=_liq)
    df["_vol_regime_enc"] = ml_features.encode_volatility_regime(_vol_reg)
    df["_macro_sens"]     = ml_features.compute_macro_sensitivity(
        _close, open_=_open,
        vix_regime=_vix_r if not _vix_r.empty else None,
        spx_futures_close=indicators.get("spx_futures_close"),
    )
    df["_macro_flag"]    = ml_features.macro_override_flag(
        df.index, close=_close, open_=_open,
        vix_regime=_vix_r if not _vix_r.empty else None,
        volatility_regime=_vol_reg,
    )
    df["_overnight_w"]   = ml_features.compute_overnight_return_weighted(_open, _close)

    # ── Broader context features on the FULL frame ───────────────────────────
    _stale_days = int(getattr(_settings, "LONG_CONTEXT_STALENESS_MAX_DAYS", 5))
    _vix_level  = indicators.get("vix_level", pd.Series(dtype=float))
    _vix_proxy  = _vix_level if not _vix_level.empty else _close

    _ctx_ts, _ctx_tm = ml_features.compute_vix_term_structure(
        _vix_proxy,
        vix_short_close=ctx.get("vix_short_close"),
        vix_long_close=ctx.get("vix_long_close"),
        staleness_max_days=_stale_days,
    )
    _ctx_cs, _ctx_cm = ml_features.compute_credit_stress(
        df.index,
        hy_close=ctx.get("credit_hy_close"),
        ig_close=ctx.get("credit_ig_close"),
        staleness_max_days=_stale_days,
    )
    _ctx_bs, _ctx_bm = ml_features.compute_market_breadth(
        df.index,
        breadth_close=ctx.get("breadth_close"),
        staleness_max_days=_stale_days,
    )
    _ctx_rs, _ctx_rm = ml_features.compute_rates_level(
        df.index,
        rates_close=ctx.get("rates_close"),
        staleness_max_days=_stale_days,
    )

    # Store as columns so they survive the meaningful-move filter below
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

    # ── Build X_19 via LongTrendModel.build_features ─────────────────────────
    # LONG_BROADER_CONTEXT_ENABLED is False, so build_features produces 19 cols
    trimmed_ind = {
        k: v.reindex(df.index) if isinstance(v, pd.Series) else v
        for k, v in indicators.items()
    }
    model = LongTrendModel()
    X_19, weights_raw, valid_pos = model.build_features(df, trimmed_ind)
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
                row.append(val if math.isfinite(val) else (0.5 if "score" in col or "slope" in col or "norm" in col else 1.0))
            except Exception:
                row.append(1.0)  # missing fallback
        ctx_rows.append(row)
    X_ctx = np.array(ctx_rows, dtype=np.float32)
    X_27 = np.hstack([X_19, X_ctx])

    # ── Timestamps for metadata ───────────────────────────────────────────────
    timestamps = df.index[valid_pos]

    # ── Normalize sample weights ──────────────────────────────────────────────
    # Balance classes then normalize to mean=1.
    class_counts = np.bincount(y.astype(int), minlength=2)
    class_w = np.ones(2, dtype=np.float32)
    for cid, cnt in enumerate(class_counts):
        if cnt > 0:
            class_w[cid] = len(y) / (2.0 * cnt)
    weights = weights_raw * class_w[y.astype(int)]
    mw = float(weights.mean())
    if mw > 0:
        weights = weights / mw

    positive_rate = float(y.mean())
    majority_baseline = max(positive_rate, 1.0 - positive_rate)

    return X_19, X_27, y, weights, timestamps, majority_baseline


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_factory() -> Callable:
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Broader-context feature ablation")
    parser.add_argument("--db",  default="novacycle.db", help="SQLite DB path")
    parser.add_argument("--yf",  action="store_true",    help="Use yfinance instead of DB")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="Output JSON path")
    parser.add_argument("--horizon",   type=int,   default=21,   help="Label horizon days")
    parser.add_argument("--threshold", type=float, default=0.02, help="Meaningful-move threshold")
    parser.add_argument("--splits",    type=int,   default=5,    help="Walk-forward folds")
    args = parser.parse_args()

    run_ts = datetime.now(tz=timezone.utc).isoformat()
    oos_gate = float(getattr(_settings, "LONG_MIN_OOS_ACCURACY_LIFT", 0.0))

    print("=" * 72)
    print("  NovaCycle — Broader-Context Feature Ablation")
    print("=" * 72)
    print(f"  Horizon:   {args.horizon} trading days")
    print(f"  Threshold: {args.threshold:.1%} meaningful move")
    print(f"  Folds:     {args.splits}")
    print(f"  OOS gate:  LONG_MIN_OOS_ACCURACY_LIFT = {oos_gate:+.4f}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    ctx: dict[str, pd.Series] = {}
    if args.yf:
        print("  Loading data from yfinance …")
        voo, vix, spx, ctx = _load_from_yfinance()
    else:
        db_path = args.db
        if not Path(db_path).exists():
            db_path = str(BACKEND / args.db)
        print(f"  Loading data from DB: {db_path} …")
        voo, vix, spx = _load_from_db(db_path)
        # DB does not currently store context candles; warn so the reader
        # understands the broader-context arm will use proxy fallbacks.
        print("  ⚠  DB load: context tickers (VIX9D, VIX3M, HYG, LQD, NYAD, TNX)")
        print("     not stored in the DB — broader-context arm uses proxy/neutral")
        print("     fallbacks.  Use --yf for a full real-data ablation.")

    print(f"  VOO rows: {len(voo)}  VIX rows: {len(vix)}")
    print()

    # ── Build feature matrices ────────────────────────────────────────────────
    print("  Building feature matrices …")
    try:
        X_19, X_27, y, weights, timestamps, majority_baseline = _build_matrices(
            voo, vix, spx, ctx,
            horizon=args.horizon,
            threshold=args.threshold,
        )
    except Exception as exc:
        print(f"\n❌ Feature build failed: {exc}")
        return 1

    n_rows = len(y)
    positive_rate = float(y.mean())
    print(f"  Labeled rows:      {n_rows}")
    print(f"  Positive rate:     {positive_rate:.3f}")
    print(f"  Majority baseline: {majority_baseline:.4f}")
    print(f"  Date range:        {timestamps[0].date()} → {timestamps[-1].date()}")
    print()

    # ── Walk-forward OOS evaluation: identical folds for both arms ────────────
    embargo = max(args.horizon, 21)
    factory = _xgb_factory()

    print(f"  Running 19-feature walk-forward ({args.splits} folds, embargo={embargo}) …")
    metrics_19, _, _ = walk_forward_evaluate(
        X_19, y, weights, model_factory=factory,
        n_splits=args.splits, embargo=embargo,
    )
    factory27 = _xgb_factory()
    print(f"  Running 27-feature walk-forward ({args.splits} folds, embargo={embargo}) …")
    metrics_27, _, _ = walk_forward_evaluate(
        X_27, y, weights, model_factory=factory27,
        n_splits=args.splits, embargo=embargo,
    )

    # ── Extract key metrics ───────────────────────────────────────────────────
    if not metrics_19.get("evaluated"):
        print(f"\n❌ 19-feature evaluation failed: {metrics_19.get('reason')}")
        return 1
    if not metrics_27.get("evaluated"):
        print(f"\n❌ 27-feature evaluation failed: {metrics_27.get('reason')}")
        return 1

    acc_19   = float(metrics_19["oos_accuracy"])
    acc_27   = float(metrics_27["oos_accuracy"])
    bal_19   = metrics_19.get("oos_balanced_accuracy")
    bal_27   = metrics_27.get("oos_balanced_accuracy")
    lift_19  = acc_19 - majority_baseline
    lift_27  = acc_27 - majority_baseline
    delta    = acc_27 - acc_19  # positive = 27-feat is better
    bal_delta = (
        float(bal_27) - float(bal_19)
        if bal_19 is not None and bal_27 is not None
        else None
    )

    # ── Gate check ────────────────────────────────────────────────────────────
    # The 27-feature model must:
    #   (a) beat the majority baseline by at least oos_gate, AND
    #   (b) beat the 19-feature baseline in OOS accuracy
    passes_gate = (lift_27 >= oos_gate) and (delta > 0.0)

    # ── Full-data feature importances for the 8 context features ────────────
    # Fit one model on all labeled rows so operators can see which context
    # features drove the 27-feat result.  This is a diagnostic pass only —
    # the OOS walk-forward accuracy is the canonical gate metric.
    context_feature_importances: dict = {}
    try:
        import xgboost as xgb  # noqa: PLC0415
        full_model = xgb.XGBClassifier(
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
        full_model.fit(X_27, y, sample_weight=weights)
        all_names = _BASE_FEATURE_NAMES + _BROADER_CONTEXT_FEATURE_NAMES
        all_imps  = full_model.feature_importances_.tolist()
        context_feature_importances = {
            name: round(float(imp), 6)
            for name, imp in zip(all_names, all_imps)
            if name in _BROADER_CONTEXT_FEATURE_NAMES
        }
    except Exception as exc:
        logger.warning("Could not extract full-data feature importances: %s", exc)

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    print(f"  {'Metric':<38} {'19-feat':>9} {'27-feat':>9} {'Δ (27−19)':>10}")
    print("  " + "-" * 70)
    print(f"  {'OOS accuracy':<38} {acc_19:>9.4f} {acc_27:>9.4f} {delta:>+10.4f}")
    print(f"  {'OOS lift vs majority baseline':<38} {lift_19:>+9.4f} {lift_27:>+9.4f}")
    if bal_19 is not None and bal_27 is not None:
        print(f"  {'OOS balanced accuracy':<38} {bal_19:>9.4f} {bal_27:>9.4f} {bal_delta:>+10.4f}")
    print(f"  {'Majority baseline':<38} {majority_baseline:>9.4f}")
    print(f"  {'LONG_MIN_OOS_ACCURACY_LIFT gate':<38} {oos_gate:>+9.4f}")
    print()

    if context_feature_importances:
        print("  Context feature importances (full-data 27-feat model):")
        for feat, imp in sorted(context_feature_importances.items(), key=lambda kv: -kv[1]):
            print(f"    {feat:<35} {imp:.6f}")
        print()

    if passes_gate:
        print(f"  ✅ PASS — 27-feature model beats 19-feature baseline by {delta:+.4f}")
        print(f"     and clears the OOS gate ({lift_27:+.4f} ≥ {oos_gate:+.4f}).")
        print()
        print("  Recommendation: set LONG_BROADER_CONTEXT_ENABLED=True after a")
        print("  gate-passing retrain.  Record this delta in config.py.")
    else:
        reasons = []
        if delta <= 0.0:
            reasons.append(
                f"27-feat does NOT beat 19-feat baseline (Δ={delta:+.4f})"
            )
        if lift_27 < oos_gate:
            reasons.append(
                f"27-feat lift ({lift_27:+.4f}) < gate ({oos_gate:+.4f})"
            )
        print(f"  ❌ FAIL — do NOT enable LONG_BROADER_CONTEXT_ENABLED.")
        for r in reasons:
            print(f"     • {r}")

    print()

    # ── Save auditable JSON report ────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "ablation": "broader_context_features",
        "run_timestamp_utc": run_ts,
        "data_source": "yfinance" if args.yf else "db",
        "horizon_days": args.horizon,
        "meaningful_move_threshold": args.threshold,
        "n_splits": args.splits,
        "embargo_rows": embargo,
        "n_labeled_rows": n_rows,
        "date_range_start": str(timestamps[0].date()),
        "date_range_end":   str(timestamps[-1].date()),
        "positive_rate": round(positive_rate, 4),
        "majority_baseline": round(majority_baseline, 4),
        "LONG_MIN_OOS_ACCURACY_LIFT": oos_gate,
        "baseline_19feat": {
            "n_features": int(X_19.shape[1]),
            "feature_names": _BASE_FEATURE_NAMES,
            "oos_accuracy": round(acc_19, 4),
            "oos_lift_vs_majority": round(lift_19, 4),
            "oos_balanced_accuracy": round(float(bal_19), 4) if bal_19 is not None else None,
            "folds": metrics_19.get("folds"),
        },
        "candidate_27feat": {
            "n_features": int(X_27.shape[1]),
            "feature_names": _BASE_FEATURE_NAMES + _BROADER_CONTEXT_FEATURE_NAMES,
            "oos_accuracy": round(acc_27, 4),
            "oos_lift_vs_majority": round(lift_27, 4),
            "oos_balanced_accuracy": round(float(bal_27), 4) if bal_27 is not None else None,
            "folds": metrics_27.get("folds"),
            # Full-data feature importances for the 8 context features only.
            # Diagnostic: shows which context features the model relied on.
            # The OOS walk-forward accuracy above is the canonical gate metric.
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
        # Record for the config comment when the gate passes
        "promotion_record": (
            {
                "date": run_ts[:10],
                "accuracy_delta": round(delta, 4),
                "lift_27feat": round(lift_27, 4),
                "note": (
                    "27-feature model cleared OOS gate on this date. "
                    "Set LONG_BROADER_CONTEXT_ENABLED=True and trigger a retrain."
                ),
            }
            if passes_gate
            else None
        ),
    }

    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"  📄 Auditable report saved → {out_path}")
    print()

    return 0 if passes_gate else 2  # 0=pass, 2=fail (not error)


if __name__ == "__main__":
    sys.exit(main())
