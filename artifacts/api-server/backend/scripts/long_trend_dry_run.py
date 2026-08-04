"""
Long-Trend Feature / Target Exploration Harness
================================================
Isolated dry-run: NEVER writes to ml/models or the live database.

All model/calibration artefacts are redirected to /tmp/lt_dryrun_<pid>.
The production DB is opened read-only via the sqlite URI (file:...?mode=ro).

Usage (run from artifacts/api-server/backend/):

     python scripts/long_trend_dry_run.py [--db PATH] [--quick] [--combo H,T]
                                             [--benchmark]

    --db PATH      Override the SQLite DB path (default: novacycle.db).
    --quick        Run only the core matrix (fewer feature variants).
    --combo H,T    Run a single combination only, e.g. --combo 21,0.02.
    --yf           Fetch data from yfinance instead of the DB (offline / CI).

Exit code 0 always — this is a research tool, not a gate.

Methodology
-----------
* Data is loaded once from the DB (read-only) and cached.
* For each (horizon, threshold, feature_set, model) configuration:
    1.  Build labels: y(t)=1 if forward_return_H > T, y(t)=0 if < -T.
        Rows where |return| < T are excluded (noise).
        Labels use close.shift(-H) — strictly future prices only.
    2.  Pre-compute all rolling features on the FULL unfiltered frame BEFORE
        the noise-row filter, then reindex to the filtered subset.
        This prevents the train/inference mismatch bug (see memory note
        long-trend-return-alignment.md).
    3.  Purged chronological walk-forward OOS (n_splits=5, embargo=H rows).
    4.  Report OOS accuracy, balanced accuracy, lift vs majority baseline.

Results are printed as a markdown table and saved to a JSON summary.
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
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# ── Redirect ml.calibration.MODEL_DIR before importing ml modules ─────────────
# This prevents any accidental write to ml/models even if a code path
# inside calibration.py tries to persist something.
import ml.calibration as _cal_mod

_DRY_RUN_DIR = Path(tempfile.mkdtemp(prefix="lt_dryrun_"))
_cal_mod.MODEL_DIR = _DRY_RUN_DIR
_cal_mod.CALIBRATOR_PATH = _DRY_RUN_DIR / "long_trend_calibrator.pkl"
_cal_mod.REPORT_PATH = _DRY_RUN_DIR / "long_trend_calibration.json"

from ml.calibration import walk_forward_evaluate
from ml.long_trend import FEATURE_NAMES, VIX_REGIME_MAP, LongTrendModel
from indicators.technical import TechnicalIndicators

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("lt_dry_run")

# ─────────────────────────────────────────────────────────────────────────────
# Exploration grid
# ─────────────────────────────────────────────────────────────────────────────

HORIZONS = [5, 10, 21, 42]
THRESHOLDS = [0.0, 0.01, 0.02, 0.03]  # 0.0 = sign-only (no noise filter)

# Feature sets: (name, list of FEATURE_NAMES columns to keep, or None=all)
FEATURE_SETS = [
    ("all_19",      None),          # current full feature set
    ("no_macro",    [               # drop the 4 additive macro features
        "sma50_200_ratio", "macd_line", "macd_signal", "adx",
        "vix_regime_enc", "return_5d", "return_10d", "return_20d",
        "volume_ratio", "atr_norm", "sma20_distance",
        "vix_level_norm", "vix_change_5d", "vix_percentile_1y", "vix_missing",
    ]),
    ("momentum_vix", [              # minimal: momentum + VIX only
        "return_5d", "return_10d", "return_20d",
        "vix_regime_enc", "vix_level_norm", "vix_percentile_1y",
    ]),
    ("trend_mom",   [               # trend + momentum, no VIX
        "sma50_200_ratio", "macd_line", "macd_signal", "adx",
        "return_5d", "return_10d", "return_20d", "atr_norm", "sma20_distance",
    ]),
]

MODELS = [
    "xgboost",
    "logistic",   # calibrated LR baseline
]

# ─────────────────────────────────────────────────────────────────────────────
# sklearn wrapper: walk_forward_evaluate calls model.fit(..., verbose=False)
# which sklearn does not accept.  Wrap sklearn models to absorb that kwarg.
# ─────────────────────────────────────────────────────────────────────────────

class _SklearnWrapper:
    """Thin wrapper so sklearn classifiers absorb XGBoost-style fit kwargs."""

    def __init__(self, clf):
        self._clf = clf

    def fit(self, X, y, sample_weight=None, verbose=False, **kw):
        from sklearn.pipeline import Pipeline
        # fold-local StandardScaler to avoid data leakage (fit on train only)
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        self._pipe = make_pipeline(StandardScaler(), self._clf)
        self._pipe.fit(X, y, logisticregression__sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self._pipe.predict_proba(X)


def _xgb_factory(max_depth: int = 3) -> Callable:
    def factory():
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=max_depth,
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


def _logistic_factory() -> Callable:
    def factory():
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        return _SklearnWrapper(clf)
    return factory


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_db(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load daily VOO, VIX, and SPX candles from DB read-only.

    Returns (voo_df, vix_df, spx_series) — voo/vix indexed by timestamp.
    """
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)

    voo = pd.read_sql_query(
        """
        SELECT timestamp, open, high, low, close, volume, is_extended_hours, session_type
        FROM voo_candles
        WHERE ticker='VOO' AND timeframe='daily' AND is_extended_hours=0
        ORDER BY timestamp ASC
        """,
        con,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    vix = pd.read_sql_query(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM vix_candles
        WHERE ticker='^VIX' AND timeframe='daily'
        ORDER BY timestamp ASC
        """,
        con,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    spx_rows = pd.read_sql_query(
        """
        SELECT timestamp, close
        FROM spx_candles
        WHERE timeframe='daily'
        ORDER BY timestamp ASC
        """,
        con,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    con.close()

    voo.index = pd.to_datetime(voo.index)
    voo = voo[~voo.index.duplicated(keep="last")]
    vix.index = pd.to_datetime(vix.index)
    vix = vix[~vix.index.duplicated(keep="last")]

    spx_series = pd.Series(dtype=float)
    if not spx_rows.empty:
        spx_rows.index = pd.to_datetime(spx_rows.index)
        spx_series = spx_rows["close"]
        spx_series = spx_series[~spx_series.index.duplicated(keep="last")]

    return voo, vix, spx_series


def _load_from_yfinance(start: str = "2015-01-01") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch data from yfinance for offline / CI use."""
    import yfinance as yf

    def _fetch(ticker, cols):
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
    spx = _fetch("ES=F", ["close"])
    spx_series = spx["close"].dropna()

    return voo, vix, spx_series


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder (reuses LongTrendModel.build_features)
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_features(
    voo: pd.DataFrame,
    vix: pd.DataFrame,
    spx_series: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute all indicators and add pre-computed temporal columns to voo,
    then return (enriched_voo, indicators).

    Pre-computation on the FULL unfiltered frame is the critical anti-leakage
    step: rolling windows span real trading days, not the filtered subset.
    """
    ti = TechnicalIndicators()
    indicators = ti.compute_all(voo, vix, exclude_extended=True)
    if not spx_series.empty:
        indicators["spx_futures_close"] = spx_series

    df = voo.copy()
    # Same pre-computation as LongTrendModel.train():
    df["_return_5d"]  = df["close"].pct_change(5)
    df["_return_10d"] = df["close"].pct_change(10)
    df["_return_20d"] = df["close"].pct_change(20)
    df["_vol_avg20"]  = df["volume"].rolling(20).mean() if "volume" in df.columns else 0.0

    import ml.features as ml_features
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

    return df, indicators


# ─────────────────────────────────────────────────────────────────────────────
# Single-configuration dry-run
# ─────────────────────────────────────────────────────────────────────────────

def run_config(
    df_full: pd.DataFrame,
    indicators_full: dict,
    horizon: int,
    threshold: float,
    feature_cols: Optional[list],
    model_type: str,
    label: str,
) -> dict:
    """
    Run one configuration and return a metrics dict.

    Args:
        df_full:         Enriched VOO frame (pre-computed _return_*, _vol_*, etc.)
        indicators_full: Full indicators dict aligned to df_full
        horizon:         Forward-return horizon in trading days
        threshold:       Meaningful-move threshold (0.0 = sign-only)
        feature_cols:    Subset of FEATURE_NAMES to keep (None = all 19)
        model_type:      "xgboost" | "logistic"
        label:           Human-readable config label for the results table
    """
    from config import settings
    import math

    df = df_full.copy()

    # ── Labels: forward return using strictly future prices ────────────────────
    df["_future_close"] = df["close"].shift(-horizon)
    df = df.dropna(subset=["_future_close"])
    df["_fwd_return"] = df["_future_close"] / df["close"] - 1.0

    if threshold > 0.0:
        df = df[(df["_fwd_return"] >= threshold) | (df["_fwd_return"] <= -threshold)].copy()

    if len(df) < 150:
        return {
            "label": label,
            "horizon": horizon,
            "threshold": threshold,
            "model": model_type,
            "features": feature_cols or "all_19",
            "error": f"too few rows after filter ({len(df)})",
        }

    df["_label"] = (df["_fwd_return"] >= (threshold if threshold > 0 else 0.0)).astype(int)

    # ── Trim indicators to the filtered df's index ────────────────────────────
    trimmed_ind = {
        k: (v.reindex(df.index) if isinstance(v, pd.Series) else v)
        for k, v in indicators_full.items()
    }

    # ── Build feature matrix via LongTrendModel.build_features ────────────────
    lt = LongTrendModel.__new__(LongTrendModel)
    lt.model = None
    X_all, weights, valid_pos = lt.build_features(df, trimmed_ind)
    y_all = df["_label"].values[valid_pos]

    if len(X_all) < 150:
        return {
            "label": label,
            "horizon": horizon,
            "threshold": threshold,
            "model": model_type,
            "features": feature_cols or "all_19",
            "error": f"too few feature rows ({len(X_all)})",
        }

    # ── Select feature subset ─────────────────────────────────────────────────
    if feature_cols is not None:
        col_idx = [FEATURE_NAMES.index(c) for c in feature_cols if c in FEATURE_NAMES]
        X_all = X_all[:, col_idx]
    n_features = X_all.shape[1]

    # ── Balance classes and normalize weights ─────────────────────────────────
    class_counts = np.bincount(y_all.astype(int), minlength=2)
    class_weights = np.ones(2, dtype=np.float32)
    for cid, cnt in enumerate(class_counts):
        if cnt > 0:
            class_weights[cid] = len(y_all) / (2.0 * cnt)
    weights = weights * class_weights[y_all.astype(int)]
    mw = float(weights.mean())
    if mw > 0:
        weights = weights / mw

    # ── Model factory ─────────────────────────────────────────────────────────
    if model_type == "xgboost":
        factory = _xgb_factory()
    else:
        factory = _logistic_factory()

    # ── Purged walk-forward OOS (embargo = horizon rows) ──────────────────────
    vix_col_idx = (
        FEATURE_NAMES.index("vix_regime_enc")
        if feature_cols is None or "vix_regime_enc" in (feature_cols or [])
        else None
    )
    regime_labels = None
    if vix_col_idx is not None and feature_cols is None:
        regime_labels = X_all[:, vix_col_idx].astype(np.intp)
    elif vix_col_idx is not None and feature_cols is not None and "vix_regime_enc" in feature_cols:
        local_idx = feature_cols.index("vix_regime_enc")
        regime_labels = X_all[:, local_idx].astype(np.intp)

    metrics, oos_probs, oos_labels = walk_forward_evaluate(
        X_all, y_all, weights,
        model_factory=factory,
        n_splits=5,
        embargo=max(horizon, 21),  # embargo >= horizon to prevent label leakage
        regime_labels=regime_labels,
    )

    positive_rate = float(y_all.mean())
    majority_baseline = max(positive_rate, 1.0 - positive_rate)

    return {
        "label": label,
        "horizon": horizon,
        "threshold": threshold,
        "model": model_type,
        "features": ",".join(feature_cols) if feature_cols else "all_19",
        "n_feature_cols": n_features,
        "n_rows": len(X_all),
        "positive_rate": round(positive_rate, 4),
        "majority_baseline": round(majority_baseline, 4),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in metrics.items()
           if k not in ("reliability_bins", "folds", "regime_breakdown")},
        "regime_breakdown": metrics.get("regime_breakdown"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Majority-class baseline (trivial: always predict majority)
# ─────────────────────────────────────────────────────────────────────────────

def majority_baseline_result(
    df_full: pd.DataFrame, horizon: int, threshold: float,
) -> dict:
    df = df_full.copy()
    df["_future_close"] = df["close"].shift(-horizon)
    df = df.dropna(subset=["_future_close"])
    df["_fwd_return"] = df["_future_close"] / df["close"] - 1.0
    if threshold > 0.0:
        df = df[(df["_fwd_return"] >= threshold) | (df["_fwd_return"] <= -threshold)].copy()
    if len(df) == 0:
        return {}
    y = (df["_fwd_return"] >= (threshold if threshold > 0 else 0.0)).astype(int)
    pos_rate = float(y.mean())
    majority = max(pos_rate, 1.0 - pos_rate)
    return {
        "label": f"MAJORITY h={horizon} t={threshold:.0%}",
        "horizon": horizon,
        "threshold": threshold,
        "model": "majority_class",
        "features": "—",
        "n_rows": len(y),
        "positive_rate": round(pos_rate, 4),
        "majority_baseline": round(majority, 4),
        "oos_accuracy": round(majority, 4),
        "oos_balanced_accuracy": 0.5,
        "accuracy_lift_vs_majority": 0.0,
        "evaluated": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Read-only strategy benchmark
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_HORIZON = 21
STRATEGY_THRESHOLD = 0.02
TRADING_DAYS_PER_YEAR = 252
VOL_TARGET_ANNUAL = 0.10


def _strategy_performance(
    positions: np.ndarray,
    next_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> dict:
    """Calculate transparent long-only portfolio metrics.

    A position observed at the close of day t is applied to the return from
    t to t+1.  Positions are therefore never allowed to use the next day's
    close.  Turnover is one-way absolute position change, annualized by the
    number of 252-day years in the evaluated window.  Downside capture is the
    strategy's cumulative return on benchmark-negative days divided by the
    absolute cumulative benchmark return on those same days.
    """
    positions = np.asarray(positions, dtype=float)
    next_returns = np.asarray(next_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    valid = np.isfinite(positions) & np.isfinite(next_returns) & np.isfinite(benchmark_returns)
    positions = positions[valid]
    next_returns = next_returns[valid]
    benchmark_returns = benchmark_returns[valid]
    if len(next_returns) == 0:
        return {
            "evaluated": False,
            "reason": "no valid next-day returns",
        }

    portfolio_returns = positions * next_returns
    equity = np.cumprod(1.0 + portfolio_returns)
    benchmark_equity = np.cumprod(1.0 + benchmark_returns)
    years = len(portfolio_returns) / TRADING_DAYS_PER_YEAR
    total_return = float(equity[-1] - 1.0)
    benchmark_total_return = float(benchmark_equity[-1] - 1.0)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else None
    daily_std = float(np.std(portfolio_returns, ddof=1)) if len(portfolio_returns) > 1 else 0.0
    sharpe = (
        float(np.mean(portfolio_returns) / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_std > 0
        else None
    )
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    max_drawdown = float(np.min(drawdown))
    turnover = float(
        (abs(positions[0]) + np.abs(np.diff(positions)).sum()) / years
        if years > 0 else 0.0
    )
    down_mask = benchmark_returns < 0
    down_benchmark = float(np.prod(1.0 + benchmark_returns[down_mask]) - 1.0) if down_mask.any() else 0.0
    down_strategy = float(np.prod(1.0 + portfolio_returns[down_mask]) - 1.0) if down_mask.any() else 0.0
    # Both values are negative during benchmark-down days; retaining the
    # denominator sign makes 1.0 mean identical downside participation and
    # values below 1.0 mean the strategy protected capital.
    downside_capture = (
        float(down_strategy / down_benchmark)
        if down_benchmark < 0
        else None
    )
    return {
        "evaluated": True,
        "n_days": int(len(portfolio_returns)),
        "total_return": total_return,
        "benchmark_total_return": benchmark_total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annualized_turnover": turnover,
        "downside_capture": downside_capture,
    }


def _classification_accuracy(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    threshold: float,
) -> Optional[float]:
    """Accuracy only on meaningful H-day labels, matching the current target."""
    predictions = np.asarray(predictions, dtype=int)
    forward_returns = np.asarray(forward_returns, dtype=float)
    meaningful = np.isfinite(forward_returns) & (
        (forward_returns >= threshold) | (forward_returns <= -threshold)
    )
    if not meaningful.any():
        return None
    labels = (forward_returns[meaningful] >= threshold).astype(int)
    return float((predictions[meaningful] == labels).mean())


def _benchmark_strategy_positions(
    close: pd.Series,
    decision_dates: pd.DatetimeIndex,
    model_predictions: Optional[np.ndarray],
) -> dict[str, np.ndarray]:
    """Build causal positions for each benchmark on decision dates."""
    close = close.sort_index()
    dates = pd.DatetimeIndex(decision_dates)
    close_at_decision = close.reindex(dates)
    returns = close.pct_change()
    next_returns = (close.shift(-1) / close - 1.0).reindex(dates).to_numpy(dtype=float)

    # The classifier is converted to a long/flat position so its financial
    # result is directly comparable with buy-and-hold and risk-off filters.
    if model_predictions is None:
        model_position = np.full(len(dates), np.nan)
    else:
        model_position = np.asarray(model_predictions, dtype=float)

    sma200 = close.rolling(200, min_periods=200).mean().reindex(dates)
    sma_position = (close_at_decision > sma200).astype(float).to_numpy()
    realized_vol = returns.rolling(20, min_periods=20).std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    vol_position = (VOL_TARGET_ANNUAL / realized_vol).clip(lower=0.0, upper=1.0).reindex(dates)
    vol_position = vol_position.fillna(0.0).to_numpy(dtype=float)

    return {
        "current_long_model": model_position,
        "always_up": np.ones(len(dates), dtype=float),
        "buy_and_hold": np.ones(len(dates), dtype=float),
        "sma200_filter": sma_position,
        "volatility_targeted": vol_position,
        "_next_returns": next_returns,
        "_benchmark_returns": next_returns,
    }


def benchmark_current_long_model(
    df_full: pd.DataFrame,
    indicators_full: dict,
    horizon: int = STRATEGY_HORIZON,
    threshold: float = STRATEGY_THRESHOLD,
) -> dict:
    """Benchmark the current long model and simple strategies out of sample.

    Unlike ``run_config``, this function does not remove test rows whose future
    return is small.  It trains each fold only on past meaningful labels, then
    predicts every valid out-of-sample decision day.  This makes the financial
    simulation causal and prevents a future-return filter from deciding when a
    strategy is allowed to trade.
    """
    df = df_full.sort_index().copy()
    df["_future_close"] = df["close"].shift(-horizon)
    df["_forward_return"] = df["_future_close"] / df["close"] - 1.0

    lt = LongTrendModel.__new__(LongTrendModel)
    lt.model = None
    X_all, weights, valid_pos = lt.build_features(df, indicators_full)
    if len(X_all) < 150:
        return {"evaluated": False, "reason": f"too few feature rows ({len(X_all)})"}

    feature_dates = pd.DatetimeIndex(df.index[valid_pos])
    forward_returns = df["_forward_return"].to_numpy(dtype=float)[valid_pos]
    next_returns_all = (df["close"].shift(-1) / df["close"] - 1.0).to_numpy(dtype=float)[valid_pos]
    label_mask = np.isfinite(forward_returns) & (
        (forward_returns >= threshold) | (forward_returns <= -threshold)
    )
    y_all = (forward_returns >= threshold).astype(int)

    n = len(X_all)
    embargo = max(horizon, 21)
    min_train = max(100, embargo * 3)
    test_start = max(min_train + embargo, int(n * 0.5))
    fold_edges = np.linspace(test_start, n, 6, dtype=int)
    fold_results: list[dict] = []
    pooled_positions: dict[str, list[np.ndarray]] = {}
    pooled_returns: list[np.ndarray] = []
    pooled_benchmark_returns: list[np.ndarray] = []
    pooled_model_predictions: list[np.ndarray] = []
    pooled_sma_predictions: list[np.ndarray] = []
    pooled_forward_returns: list[np.ndarray] = []

    for fold_number in range(5):
        t0, t1 = int(fold_edges[fold_number]), int(fold_edges[fold_number + 1])
        train_end = t0 - embargo
        train_mask = np.arange(n) < train_end
        train_mask &= label_mask
        if train_mask.sum() < min_train or t1 <= t0:
            continue

        y_train = y_all[train_mask]
        train_weights = weights[train_mask].copy()
        counts = np.bincount(y_train.astype(int), minlength=2)
        class_weights = np.ones(2, dtype=np.float32)
        for class_id, count in enumerate(counts):
            if count > 0:
                class_weights[class_id] = len(y_train) / (2.0 * count)
        train_weights *= class_weights[y_train.astype(int)]
        if train_weights.mean() > 0:
            train_weights /= train_weights.mean()

        model = _xgb_factory()()
        model.fit(X_all[train_mask], y_train, sample_weight=train_weights, verbose=False)
        probs = model.predict_proba(X_all[t0:t1])[:, 1]
        predictions = (probs >= 0.5).astype(int)
        dates = feature_dates[t0:t1]
        benchmark = _benchmark_strategy_positions(
            df["close"], dates, predictions.astype(float)
        )
        fold_metrics = {
            "fold": fold_number + 1,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(t1 - t0),
            "model_accuracy": _classification_accuracy(
                predictions, forward_returns[t0:t1], threshold
            ),
            "always_up_accuracy": _classification_accuracy(
                np.ones(t1 - t0, dtype=int), forward_returns[t0:t1], threshold
            ),
            "sma200_accuracy": _classification_accuracy(
                benchmark["sma200_filter"].astype(int), forward_returns[t0:t1], threshold
            ),
            "strategies": {},
        }
        fold_next_returns = benchmark["_next_returns"]
        fold_benchmark_returns = benchmark["_benchmark_returns"]
        for name in ("current_long_model", "always_up", "buy_and_hold", "sma200_filter", "volatility_targeted"):
            fold_metrics["strategies"][name] = _strategy_performance(
                benchmark[name], fold_next_returns, fold_benchmark_returns
            )
            pooled_positions.setdefault(name, []).append(benchmark[name])
        pooled_returns.append(fold_next_returns)
        pooled_benchmark_returns.append(fold_benchmark_returns)
        pooled_model_predictions.append(predictions)
        pooled_sma_predictions.append(benchmark["sma200_filter"].astype(int))
        pooled_forward_returns.append(forward_returns[t0:t1])
        fold_results.append(fold_metrics)

    if not fold_results:
        return {"evaluated": False, "reason": "no valid walk-forward folds"}

    pooled_next_returns = np.concatenate(pooled_returns)
    pooled_benchmark = np.concatenate(pooled_benchmark_returns)
    pooled_forward = np.concatenate(pooled_forward_returns)
    pooled_predictions = np.concatenate(pooled_model_predictions)
    pooled_sma = np.concatenate(pooled_sma_predictions)
    pooled = {}
    for name, position_parts in pooled_positions.items():
        pooled[name] = _strategy_performance(
            np.concatenate(position_parts), pooled_next_returns, pooled_benchmark
        )

    return {
        "evaluated": True,
        "method": "causal_purged_walk_forward_strategy_benchmark",
        "horizon": horizon,
        "threshold": threshold,
        "embargo_rows": embargo,
        "volatility_target_annual": VOL_TARGET_ANNUAL,
        "n_oos_decision_days": int(len(pooled_next_returns)),
        "pooled_accuracy": {
            "current_long_model": _classification_accuracy(
                pooled_predictions, pooled_forward, threshold
            ),
            "always_up": _classification_accuracy(
                np.ones(len(pooled_forward), dtype=int), pooled_forward, threshold
            ),
            "sma200_filter": _classification_accuracy(
                pooled_sma, pooled_forward, threshold
            ),
        },
        "pooled_strategies": pooled,
        "folds": fold_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results table printer
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, width=8):
    if v is None:
        return "—".center(width)
    if isinstance(v, bool):
        return str(v).center(width)
    if isinstance(v, float):
        return f"{v:+.4f}".center(width)
    return str(v).center(width)


def print_results_table(results: list[dict]) -> None:
    cols = [
        ("Config / Label",   40, "label"),
        ("H",                 4, "horizon"),
        ("T%",                5, "threshold"),
        ("Model",            10, "model"),
        ("N",                 6, "n_rows"),
        ("pos%",              6, "positive_rate"),
        ("OOS acc",           9, "oos_accuracy"),
        ("Bal acc",           9, "oos_balanced_accuracy"),
        ("Lift",              9, "accuracy_lift_vs_majority"),
        ("Maj base",          9, "majority_baseline"),
        ("OK?",               6, None),
    ]

    def row_line(r):
        parts = []
        for name, w, key in cols:
            if key is None:
                lift = r.get("accuracy_lift_vs_majority")
                ok = "✓" if (lift is not None and lift > 0) else "✗"
                parts.append(ok.center(w))
            elif key == "threshold":
                t = r.get("threshold", 0.0)
                parts.append(f"{t:.0%}".center(w))
            elif isinstance(r.get(key), float):
                parts.append(_fmt(r[key], w))
            else:
                v = str(r.get(key, ""))
                parts.append(v[:w].ljust(w))
        return " | ".join(parts)

    header = " | ".join(name[:w].center(w) for name, w, _ in cols)
    sep = "-+-".join("-" * w for name, w, _ in cols)
    print()
    print(header)
    print(sep)
    for r in results:
        print(row_line(r))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Long-trend dry-run exploration harness")
    parser.add_argument("--db", default="novacycle.db", help="SQLite DB path")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer configs (baseline H/T grid + all_19 only)")
    parser.add_argument("--combo", default=None,
                        help="Run single combo only, e.g. --combo 21,0.02")
    parser.add_argument("--yf", action="store_true",
                        help="Fetch data from yfinance instead of DB")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also run the causal strategy benchmark for the current long target",
    )
    args = parser.parse_args()

    # ── Guard: confirm no writes to ml/models ─────────────────────────────────
    real_models_dir = BACKEND / "ml" / "models"
    assert str(_DRY_RUN_DIR) != str(real_models_dir), "Dry-run dir collision!"
    print(f"\n🧪 Dry-run temp dir : {_DRY_RUN_DIR}")
    print(f"   Real models dir  : {real_models_dir}  (untouched)")

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.yf:
        print("\n📡 Fetching data from yfinance …")
        voo, vix, spx_series = _load_from_yfinance()
    else:
        db_path = args.db if os.path.isabs(args.db) else str(BACKEND / args.db)
        if not Path(db_path).exists():
            print(f"ERROR: DB not found at {db_path}. Use --yf for yfinance mode.")
            sys.exit(1)
        print(f"\n📂 Loading from DB (read-only): {db_path}")
        voo, vix, spx_series = _load_from_db(db_path)

    print(f"   VOO rows         : {len(voo)}  ({voo.index[0].date()} → {voo.index[-1].date()})")
    print(f"   VIX rows         : {len(vix)}")
    print(f"   SPX rows         : {len(spx_series)}")

    if len(voo) < 200:
        print("ERROR: Not enough VOO data (need ≥200 daily rows). Aborting.")
        sys.exit(1)

    # ── Build full enriched feature frame once ────────────────────────────────
    print("\n⚙️  Computing indicators and pre-computing temporal features …")
    df_full, indicators_full = _build_full_features(voo, vix, spx_series)
    print(f"   Frame columns    : {list(df_full.columns)[:8]} …")

    # ── Build config list ─────────────────────────────────────────────────────
    if args.combo:
        h_str, t_str = args.combo.split(",")
        configs = [(int(h_str), float(t_str))]
        feature_grid = FEATURE_SETS
        model_grid = MODELS
    elif args.quick:
        configs = [(h, t) for h in HORIZONS for t in [0.0, 0.02]]
        feature_grid = [("all_19", None)]
        model_grid = ["xgboost", "logistic"]
    else:
        configs = [(h, t) for h in HORIZONS for t in THRESHOLDS]
        feature_grid = FEATURE_SETS
        model_grid = MODELS

    # ── Run majority baselines first ──────────────────────────────────────────
    results: list[dict] = []
    seen_baselines: set = set()
    for h, t in configs:
        key = (h, t)
        if key not in seen_baselines:
            seen_baselines.add(key)
            r = majority_baseline_result(df_full, h, t)
            if r:
                results.append(r)

    # ── Run all configurations ────────────────────────────────────────────────
    total = len(configs) * len(feature_grid) * len(model_grid)
    done = 0
    for h, t in configs:
        for fs_name, fs_cols in feature_grid:
            for mdl in model_grid:
                done += 1
                lbl = f"h={h} t={t:.0%} feat={fs_name} mdl={mdl}"
                print(f"\r[{done}/{total}] {lbl:<60}", end="", flush=True)
                t0 = time.time()
                r = run_config(df_full, indicators_full, h, t, fs_cols, mdl, lbl)
                r["elapsed_s"] = round(time.time() - t0, 1)
                results.append(r)

    print("\n")

    # ── Print table ───────────────────────────────────────────────────────────
    print_results_table(results)

    # ── Run causal financial benchmark when requested ─────────────────────────
    benchmark_result = None
    if args.benchmark:
        from config import settings
        benchmark_horizon = settings.LONG_LABEL_HORIZON_DAYS
        benchmark_threshold = settings.LONG_MEANINGFUL_MOVE_THRESHOLD
        if args.combo:
            h_str, t_str = args.combo.split(",")
            benchmark_horizon = int(h_str)
            benchmark_threshold = float(t_str)
        print(
            "\n📈 Running causal strategy benchmark "
            f"(h={benchmark_horizon}, threshold={benchmark_threshold:.0%}) …"
        )
        benchmark_result = benchmark_current_long_model(
            df_full,
            indicators_full,
            horizon=benchmark_horizon,
            threshold=benchmark_threshold,
        )
        if benchmark_result.get("evaluated"):
            print(
                f"   OOS decision days: {benchmark_result['n_oos_decision_days']}; "
                f"folds: {len(benchmark_result['folds'])}"
            )
            for name, metrics in benchmark_result["pooled_strategies"].items():
                if metrics.get("evaluated"):
                    print(
                        f"   {name:<22} "
                        f"return={metrics.get('total_return', float('nan')):+.2%} "
                        f"CAGR={metrics.get('cagr', float('nan')):+.2%} "
                        f"Sharpe={metrics.get('sharpe', float('nan')):+.2f} "
                        f"maxDD={metrics.get('max_drawdown', float('nan')):+.2%} "
                        f"turnover={metrics.get('annualized_turnover', float('nan')):.2f} "
                        f"downside={metrics.get('downside_capture', float('nan'))}"
                    )
        else:
            print(f"   Benchmark unavailable: {benchmark_result.get('reason')}")

    # ── Confirm no writes to real models dir ──────────────────────────────────
    real_model_files = list(real_models_dir.glob("*.pkl")) + list(real_models_dir.glob("*.json"))
    print(f"✅ Real models dir file count unchanged: {len(real_model_files)} files")
    print(f"✅ Dry-run artefacts in: {_DRY_RUN_DIR}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    # Save to /tmp (never to the codebase)
    summary_path = _DRY_RUN_DIR / "results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 JSON results saved to: {summary_path}")
    if benchmark_result is not None:
        benchmark_path = _DRY_RUN_DIR / "strategy_benchmark.json"
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_result, f, indent=2, default=str)
        print(f"📄 Strategy benchmark saved to: {benchmark_path}")

    # ── Print best configurations (positive lift) ─────────────────────────────
    positive = [
        r for r in results
        if isinstance(r.get("accuracy_lift_vs_majority"), float)
        and r["accuracy_lift_vs_majority"] > 0.0
        and r.get("model") not in ("majority_class",)
        and r.get("evaluated") is True
    ]
    positive.sort(key=lambda r: r.get("accuracy_lift_vs_majority", 0.0), reverse=True)

    print("\n🏆 Configurations with positive OOS lift (sorted by lift):")
    if positive:
        for r in positive[:10]:
            lift = r["accuracy_lift_vs_majority"]
            bal = r.get("oos_balanced_accuracy", "—")
            bal_s = f"{bal:+.4f}" if isinstance(bal, float) else str(bal)
            print(f"   {r['label']:<60}  lift={lift:+.4f}  bal={bal_s}")
    else:
        print("   NONE — no configuration beats the majority baseline OOS.")

    print(f"\n📊 Summary: {len(positive)}/{len(results)} configs beat the majority baseline.\n")

    return results


if __name__ == "__main__":
    main()
