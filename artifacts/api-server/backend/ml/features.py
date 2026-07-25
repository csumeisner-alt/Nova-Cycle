"""
NovaCycle ML Feature Engineering (VOO only)
===========================================
Shared, in-memory feature helpers used by both the long-trend and
short-trend ML pipelines. Nothing here touches the database schema,
endpoints, or ingestion — features are computed on the fly from the
DataFrames/indicators the models already receive.

Features provided:
  - volatility_regime       : calm | trending | macro_shock | compressed
                              (ATR + return std-dev + liquidity class)
  - macro_sensitivity_score : [0, 1] from VIX regime, SPX futures
                              (graceful fallback when unavailable) and
                              overnight returns
  - macro_override_flag     : placeholder (always 0.0 for now)
  - gap_momentum            : gap_percent × direction_of_first_candle
  - gap_momentum_class      : weak | medium | strong
  - liquidity_compression_score : volume-deviation based, reusing
                              ingestion-derived fields when present
  - overnight_return_weighted   : exponentially-weighted overnight return

All computations are vectorized and every helper falls back to safe
default values (never raises) with structured log messages.
"""

import logging

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ── Encodings (stable — order matters for saved models) ──────────────────────
VOLATILITY_REGIME_MAP = {"calm": 0, "compressed": 1, "trending": 2, "macro_shock": 3}
GAP_MOMENTUM_CLASS_MAP = {"weak": 0, "medium": 1, "strong": 2}

# Defaults used on any computation failure
DEFAULT_VOL_REGIME_ENC = float(VOLATILITY_REGIME_MAP["calm"])
DEFAULT_MACRO_SENSITIVITY = 0.5
DEFAULT_MACRO_OVERRIDE_FLAG = 0.0
DEFAULT_GAP_MOMENTUM = 0.0
DEFAULT_GAP_MOMENTUM_CLASS_ENC = float(GAP_MOMENTUM_CLASS_MAP["weak"])
DEFAULT_LIQ_COMPRESSION = 0.0
DEFAULT_OVERNIGHT_WEIGHTED = 0.0

VIX_REGIME_SCORE = {"LOW": 0.1, "NORMAL": 0.4, "HIGH": 0.7, "EXTREME": 1.0}


def _default_series(index, value: float) -> pd.Series:
    return pd.Series(value, index=index, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Volatility regime
# ─────────────────────────────────────────────────────────────────────────────

def compute_volatility_regime(
    close: pd.Series,
    atr: pd.Series | None = None,
    liquidity_class: pd.Series | None = None,
    window: int = 20,
    baseline_window: int = 100,
) -> pd.Series:
    """
    Classify each bar's volatility regime (vectorized):

      ratio      = rolling_std(returns, window) / rolling_median(std, baseline)
      atr_norm   = ATR / close (rising ATR → trending)

      macro_shock : ratio > 2.0
      compressed  : ratio < 0.5, or liquidity_class in ('thin', 'none')
      trending    : atr_norm > its rolling mean (sustained expansion)
      calm        : otherwise

    Returns a Series of regime labels; on failure, all-'calm' with a log.
    """
    try:
        ret_std = close.pct_change().rolling(window).std()
        baseline = ret_std.rolling(baseline_window, min_periods=window).median()
        ratio = (ret_std / baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        if atr is not None and not atr.empty:
            atr_norm = (atr.reindex(close.index) / close).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            atr_trend = atr_norm > atr_norm.rolling(window).mean().fillna(atr_norm)
        else:
            atr_trend = pd.Series(False, index=close.index)

        thin_liq = pd.Series(False, index=close.index)
        if liquidity_class is not None and not liquidity_class.empty:
            thin_liq = (
                liquidity_class.reindex(close.index)
                .astype(str)
                .isin(["thin", "none"])
            )

        regimes = np.select(
            [
                ratio > 2.0,
                (ratio < 0.5) | thin_liq,
                atr_trend,
            ],
            ["macro_shock", "compressed", "trending"],
            default="calm",
        )
        return pd.Series(regimes, index=close.index)
    except Exception as exc:
        logger.error("ml_feature_error feature=volatility_regime error=%s", exc)
        return pd.Series("calm", index=close.index)


def encode_volatility_regime(regimes: pd.Series) -> pd.Series:
    """Label-encode volatility regimes; unknown labels map to 'calm'."""
    try:
        return regimes.map(VOLATILITY_REGIME_MAP).fillna(DEFAULT_VOL_REGIME_ENC).astype(float)
    except Exception as exc:
        logger.error("ml_feature_error feature=volatility_regime_enc error=%s", exc)
        return _default_series(regimes.index, DEFAULT_VOL_REGIME_ENC)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Macro sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_macro_sensitivity(
    close: pd.Series,
    open_: pd.Series | None = None,
    vix_regime: pd.Series | None = None,
    spx_futures_close: pd.Series | None = None,
    window: int = 10,
) -> pd.Series:
    """
    macro_sensitivity_score in [0, 1] (vectorized), averaging:

      - VIX component     : regime mapped to [0.1 … 1.0] (forward-filled onto
                            the VOO index); 0.4 (NORMAL) when unavailable
      - Futures component : |SPX futures overnight return| rolling mean,
                            scaled by 2% cap; falls back to the VOO
                            overnight-return proxy when futures data is
                            unavailable (logged once per call)
      - Overnight component: |VOO overnight return| rolling mean, 2% cap

    On failure returns all-0.5 with a structured log.
    """
    try:
        idx = close.index

        # VIX component
        if vix_regime is not None and not vix_regime.empty:
            vix_comp = (
                vix_regime.astype(str).map(VIX_REGIME_SCORE)
                .reindex(idx, method="ffill")
                .fillna(0.4)
            )
        else:
            vix_comp = _default_series(idx, 0.4)

        # Overnight returns (VOO)
        if open_ is not None and not open_.empty:
            overnight = ((open_ - close.shift(1)) / close.shift(1)).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        else:
            overnight = close.pct_change().fillna(0.0)
        overnight_comp = (
            overnight.abs().rolling(window, min_periods=1).mean() / 0.02
        ).clip(0.0, 1.0)

        # SPX futures component (graceful fallback)
        futures_comp = None
        if spx_futures_close is not None and not spx_futures_close.empty:
            try:
                fut_ret = spx_futures_close.pct_change()
                if not fut_ret.index.equals(idx):
                    fut_ret = fut_ret.reindex(idx, method="ffill")
                fut_ret = fut_ret.fillna(0.0)
                futures_comp = (
                    fut_ret.abs().rolling(window, min_periods=1).mean() / 0.02
                ).clip(0.0, 1.0)
            except Exception as align_exc:
                logger.warning(
                    "ml_feature_fallback feature=macro_sensitivity_score "
                    "reason=spx_futures_alignment_failed error=%s "
                    "using=voo_overnight_proxy", align_exc,
                )
                futures_comp = None
        if futures_comp is None:
            logger.info(
                "ml_feature_fallback feature=macro_sensitivity_score "
                "reason=spx_futures_unavailable using=voo_overnight_proxy"
            )
            futures_comp = overnight_comp

        score = ((vix_comp + futures_comp + overnight_comp) / 3.0).clip(0.0, 1.0)
        return score.fillna(DEFAULT_MACRO_SENSITIVITY)
    except Exception as exc:
        logger.error("ml_feature_error feature=macro_sensitivity_score error=%s", exc)
        return _default_series(close.index, DEFAULT_MACRO_SENSITIVITY)


def macro_override_flag(index) -> pd.Series:
    """Placeholder macro override flag (always 0.0 for now; in-memory only)."""
    return _default_series(index, DEFAULT_MACRO_OVERRIDE_FLAG)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gap momentum
# ─────────────────────────────────────────────────────────────────────────────

def compute_gap_momentum_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    gap_momentum = gap_percent × direction_of_first_candle (per trading day)

    direction_of_first_candle = sign(close − open) of the day's first bar.

    gap_momentum_class (encoded weak=0, medium=1, strong=2):
      weak   : |gap_momentum| <  MICRO_GAP_THRESHOLD (default 0.1)
      strong : |gap_momentum| >  MACRO_GAP_THRESHOLD (default 1.0)
      medium : otherwise

    Returns (gap_momentum, gap_momentum_class_enc); zeros/weak on failure.
    """
    try:
        idx = df.index
        if "gap_percent" not in df.columns or df.empty:
            return (
                _default_series(idx, DEFAULT_GAP_MOMENTUM),
                _default_series(idx, DEFAULT_GAP_MOMENTUM_CLASS_ENC),
            )

        gap = pd.to_numeric(df["gap_percent"], errors="coerce").fillna(0.0)

        days = pd.DatetimeIndex(idx).normalize()
        first_open = df["open"].groupby(days).transform("first")
        first_close = df["close"].groupby(days).transform("first")
        direction = np.sign(first_close - first_open)

        gap_momentum = (gap * direction).fillna(0.0)

        mag = gap_momentum.abs()
        cls = np.select(
            [mag > settings.MACRO_GAP_THRESHOLD, mag >= settings.MICRO_GAP_THRESHOLD],
            [GAP_MOMENTUM_CLASS_MAP["strong"], GAP_MOMENTUM_CLASS_MAP["medium"]],
            default=GAP_MOMENTUM_CLASS_MAP["weak"],
        )
        return gap_momentum, pd.Series(cls, index=idx, dtype=float)
    except Exception as exc:
        logger.error("ml_feature_error feature=gap_momentum error=%s", exc)
        return (
            _default_series(df.index, DEFAULT_GAP_MOMENTUM),
            _default_series(df.index, DEFAULT_GAP_MOMENTUM_CLASS_ENC),
        )


def classify_gap_momentum(value: float) -> str:
    """Human-readable gap momentum class for a single value."""
    mag = abs(value)
    if mag > settings.MACRO_GAP_THRESHOLD:
        return "strong"
    if mag >= settings.MICRO_GAP_THRESHOLD:
        return "medium"
    return "weak"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Liquidity compression
# ─────────────────────────────────────────────────────────────────────────────

def compute_liquidity_compression_score(
    df: pd.DataFrame, window: int = 20
) -> pd.Series:
    """
    liquidity_compression_score in [0, 1]: how far current volume sits
    below its rolling average (1 = fully compressed, 0 = at/above average).

    Reuses the ingestion-derived `liquidity_compression` column when present
    (inverted: ingestion's value is 1.0 for healthy liquidity), instead of
    recomputing from raw volume.
    """
    try:
        if "liquidity_compression" in df.columns:
            ing = pd.to_numeric(df["liquidity_compression"], errors="coerce")
            if ing.notna().any():
                return (1.0 - ing.clip(0.0, 1.0)).fillna(DEFAULT_LIQ_COMPRESSION)

        if "volume" not in df.columns or df.empty:
            return _default_series(df.index, DEFAULT_LIQ_COMPRESSION)

        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        avg = vol.rolling(window, min_periods=1).mean()
        deviation = ((avg - vol) / avg.replace(0, np.nan)).clip(0.0, 1.0)
        return deviation.fillna(DEFAULT_LIQ_COMPRESSION)
    except Exception as exc:
        logger.error("ml_feature_error feature=liquidity_compression_score error=%s", exc)
        return _default_series(df.index, DEFAULT_LIQ_COMPRESSION)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Overnight-return weighting (long-trend)
# ─────────────────────────────────────────────────────────────────────────────

def compute_overnight_return_weighted(
    open_: pd.Series, close: pd.Series, span: int = 20
) -> pd.Series:
    """
    overnight_return_weighted: exponentially-weighted mean of overnight
    returns ((open − prev_close) / prev_close), emphasizing recent gaps
    for the long-trend model. Zeros on failure.
    """
    try:
        overnight = ((open_ - close.shift(1)) / close.shift(1)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        return overnight.ewm(span=span, adjust=False).mean().fillna(
            DEFAULT_OVERNIGHT_WEIGHTED
        )
    except Exception as exc:
        logger.error("ml_feature_error feature=overnight_return_weighted error=%s", exc)
        return _default_series(close.index, DEFAULT_OVERNIGHT_WEIGHTED)
