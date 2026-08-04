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
  - macro_override_flag     : 1.0 during genuine macro shocks (VIX regime
                              + overnight move, or volatility_regime ==
                              macro_shock); thresholds in config.py
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


def macro_override_flag(
    index,
    close: pd.Series | None = None,
    open_: pd.Series | None = None,
    vix_regime: pd.Series | None = None,
    volatility_regime: pd.Series | None = None,
) -> pd.Series:
    """
    macro_override_flag ∈ {0.0, 1.0}: flips to 1.0 during genuine macro
    shock conditions so models can learn to discount normal signals.

    Fires when either (vectorized):
      - volatility_regime == 'macro_shock', or
      - VIX regime >= settings.MACRO_OVERRIDE_VIX_REGIME (default HIGH)
        AND |overnight return| > settings.MACRO_OVERRIDE_OVERNIGHT_MOVE_PCT %

    In-memory only; all-0.0 when inputs are missing or on failure.
    """
    try:
        flag = pd.Series(False, index=index)

        # Volatility-regime condition
        if volatility_regime is not None and not volatility_regime.empty:
            flag |= (
                volatility_regime.reindex(index).astype(str) == "macro_shock"
            )

        # VIX + overnight-move condition
        if (
            vix_regime is not None
            and not vix_regime.empty
            and close is not None
            and not close.empty
        ):
            threshold_regime = str(settings.MACRO_OVERRIDE_VIX_REGIME).upper()
            threshold_score = VIX_REGIME_SCORE.get(threshold_regime, 1.0)
            vix_score = (
                vix_regime.astype(str).str.upper().map(VIX_REGIME_SCORE)
                .reindex(index, method="ffill")
                .fillna(0.0)
            )
            vix_hot = vix_score >= threshold_score

            if open_ is not None and not open_.empty:
                overnight = ((open_ - close.shift(1)) / close.shift(1)).replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0.0)
            else:
                overnight = close.pct_change().fillna(0.0)
            big_move = overnight.abs() * 100.0 > settings.MACRO_OVERRIDE_OVERNIGHT_MOVE_PCT

            flag |= vix_hot & big_move

        return flag.astype(float)
    except Exception as exc:
        logger.error("ml_feature_error feature=macro_override_flag error=%s", exc)
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


# ─────────────────────────────────────────────────────────────────────────────
# 6–9. Broader market context (gated behind LONG_BROADER_CONTEXT_ENABLED)
#      Each function returns (value_series, missing_series) where
#      missing = 1.0 means the data is absent or too stale to trust.
# ─────────────────────────────────────────────────────────────────────────────

def _stale_mask(
    target_index: pd.Index,
    source_index: pd.Index,
    staleness_max_days: int,
) -> pd.Series:
    """
    Boolean Series (True = stale) for each date in target_index where the
    most recent source date at or before that date is more than
    (staleness_max_days + 2) calendar days away.

    The +2 buffer absorbs weekends, so a Friday data point is not flagged
    stale on the following Monday.  Dates that predate the earliest source
    date are always stale.  Never raises.
    """
    if source_index.empty:
        return pd.Series(True, index=target_index)
    try:
        src_sorted = pd.DatetimeIndex(source_index).sort_values()
        tgt_ts = pd.DatetimeIndex(target_index)
        tolerance = staleness_max_days + 2  # absorb weekends

        # For each target date, find the index of the last source date ≤ tgt.
        pos = src_sorted.searchsorted(tgt_ts, side="right") - 1
        before_start = pos < 0

        # Clip pos so array indexing is safe; before_start cases are overridden.
        safe_pos = np.clip(pos, 0, len(src_sorted) - 1)
        last_src = src_sorted[safe_pos]
        gap_days = (tgt_ts - last_src).days

        stale = before_start | (gap_days > tolerance)
        # stale is a numpy bool array here; wrap directly (no .values needed)
        return pd.Series(stale, index=target_index)
    except Exception as exc:
        logger.error("ml_feature_error feature=_stale_mask error=%s", exc)
        return pd.Series(False, index=target_index)


def compute_vix_term_structure(
    vix_close: pd.Series,
    vix_short_close: "pd.Series | None" = None,
    vix_long_close: "pd.Series | None" = None,
    proxy_window_short: int = 5,
    proxy_window_long: int = 20,
    staleness_max_days: int = 5,
) -> "tuple[pd.Series, pd.Series]":
    """
    VIX term-structure slope and freshness indicator (both indexed to
    vix_close.index).

    Real mode (VIX9D + VIX3M available):
      slope = vix_short / vix_long − 1
        < 0 → contango / near-term fear receding
        > 0 → backwardation / near-term spike

    Proxy mode (term data absent):
      slope = SMA(vix, short_window) / SMA(vix, long_window) − 1
      vix_term_missing = 1.0  (proxy is always treated as stale)

    Freshness: if the last source date predates any target date by more than
    staleness_max_days + 2 calendar days, that date is marked stale even when
    a series is provided.

    Returns:
        (term_slope clipped to [−1, 1], term_missing ∈ {0.0, 1.0})
    """
    try:
        idx = vix_close.index
        have_real = (
            vix_short_close is not None and not vix_short_close.empty
            and vix_long_close is not None and not vix_long_close.empty
        )

        if have_real:
            short_a = (
                vix_short_close
                .reindex(idx, method="ffill")
                .replace([np.inf, -np.inf], np.nan)
            )
            long_a = (
                vix_long_close
                .reindex(idx, method="ffill")
                .replace([np.inf, -np.inf], np.nan)
            )
            raw_slope = (short_a / long_a.replace(0, np.nan) - 1.0)
            slope = raw_slope.clip(-1.0, 1.0).fillna(0.0)
            stale = _stale_mask(idx, vix_short_close.index, staleness_max_days)
            missing = stale.astype(float)
        else:
            # Proxy: rolling SMA ratio captures short-vs-long VIX momentum
            sma_short = vix_close.rolling(proxy_window_short, min_periods=1).mean()
            sma_long  = vix_close.rolling(proxy_window_long,  min_periods=1).mean()
            raw_slope = (sma_short / sma_long.replace(0, np.nan) - 1.0)
            slope = raw_slope.clip(-1.0, 1.0).fillna(0.0)
            missing = pd.Series(1.0, index=idx)  # proxy → always stale

        return slope, missing
    except Exception as exc:
        logger.error("ml_feature_error feature=vix_term_structure error=%s", exc)
        return (
            _default_series(vix_close.index, 0.0),
            _default_series(vix_close.index, 1.0),
        )


def compute_credit_stress(
    index: pd.Index,
    hy_close: "pd.Series | None" = None,
    ig_close: "pd.Series | None" = None,
    window: int = 20,
    staleness_max_days: int = 5,
) -> "tuple[pd.Series, pd.Series]":
    """
    Credit stress score in [0, 1] and freshness indicator.

    When HYG (high-yield) and/or LQD (investment-grade) series are available:
      spread = rolling_mean(ig_return − hy_return, window)
      score  = (spread / p95_spread + 1) / 2   mapped to [0,1]
      0.5 = neutral; > 0.5 = HY underperforming → rising stress.

    When neither series is present: score = 0.5, missing = 1.0.

    Returns:
        (credit_stress_score [0,1], credit_stress_missing ∈ {0.0, 1.0})
    """
    try:
        have_hy = hy_close is not None and not hy_close.empty
        have_ig = ig_close is not None and not ig_close.empty

        if not have_hy and not have_ig:
            return (
                _default_series(index, 0.5),
                _default_series(index, 1.0),
            )

        if have_hy:
            hy_a = hy_close.reindex(index, method="ffill").replace(
                [np.inf, -np.inf], np.nan
            )
            hy_ret = hy_a.pct_change().fillna(0.0)
        else:
            hy_ret = pd.Series(0.0, index=index)

        if have_ig:
            ig_a = ig_close.reindex(index, method="ffill").replace(
                [np.inf, -np.inf], np.nan
            )
            ig_ret = ig_a.pct_change().fillna(0.0)
        else:
            ig_ret = pd.Series(0.0, index=index)

        # Spread: positive = HY underperforming IG = stress rising
        spread = (ig_ret - hy_ret).rolling(window, min_periods=1).mean()
        p95 = float(spread.abs().quantile(0.95)) or 0.01
        score = ((spread / p95).clip(-1.0, 1.0) + 1.0) / 2.0
        score = score.clip(0.0, 1.0).fillna(0.5)

        ref = hy_close.index if have_hy else ig_close.index
        stale = _stale_mask(index, ref, staleness_max_days)
        missing = stale.astype(float)
        return score, missing
    except Exception as exc:
        logger.error("ml_feature_error feature=credit_stress error=%s", exc)
        return (
            _default_series(index, 0.5),
            _default_series(index, 1.0),
        )


def compute_market_breadth(
    index: pd.Index,
    breadth_close: "pd.Series | None" = None,
    window: int = 20,
    staleness_max_days: int = 5,
) -> "tuple[pd.Series, pd.Series]":
    """
    Market breadth score in [0, 1] and freshness indicator.

    When NYSE advance-decline (NYAD) data is available:
      Uses the window-day momentum of the AD line, normalised to [0,1].
      score > 0.5 = improving breadth; < 0.5 = deteriorating breadth.

    When absent: score = 0.5, missing = 1.0.

    Returns:
        (breadth_score [0,1], breadth_missing ∈ {0.0, 1.0})
    """
    try:
        have_data = breadth_close is not None and not breadth_close.empty
        if not have_data:
            return (
                _default_series(index, 0.5),
                _default_series(index, 1.0),
            )

        aligned = breadth_close.reindex(index, method="ffill").replace(
            [np.inf, -np.inf], np.nan
        )
        momentum = aligned.diff(window).fillna(0.0)
        p95 = float(momentum.abs().quantile(0.95)) or 0.01
        score = ((momentum / p95).clip(-1.0, 1.0) + 1.0) / 2.0
        score = score.clip(0.0, 1.0).fillna(0.5)

        stale = _stale_mask(index, breadth_close.index, staleness_max_days)
        missing = stale.astype(float)
        return score, missing
    except Exception as exc:
        logger.error("ml_feature_error feature=market_breadth error=%s", exc)
        return (
            _default_series(index, 0.5),
            _default_series(index, 1.0),
        )


def compute_rates_level(
    index: pd.Index,
    rates_close: "pd.Series | None" = None,
    clip_max_pct: float = 8.0,
    staleness_max_days: int = 5,
) -> "tuple[pd.Series, pd.Series]":
    """
    10-year Treasury yield level (normalised to [0, 1]) and freshness
    indicator.

    TNX is quoted as yield × 10 (e.g. 45 = 4.5%).  The raw value is
    divided by (clip_max_pct × 10) so that an 8 % yield maps to 1.0.

    When absent: rates_level_norm = 0.5, missing = 1.0.

    Returns:
        (rates_level_norm [0,1], rates_missing ∈ {0.0, 1.0})
    """
    try:
        have_data = rates_close is not None and not rates_close.empty
        if not have_data:
            return (
                _default_series(index, 0.5),
                _default_series(index, 1.0),
            )

        aligned = (
            rates_close
            .reindex(index, method="ffill")
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(clip_max_pct * 5.0)   # neutral mid-point on failure
        )
        # TNX unit: raw value ÷ 10 = yield %; normalise by clip_max_pct
        norm = (aligned / (clip_max_pct * 10.0)).clip(0.0, 1.0)

        stale = _stale_mask(index, rates_close.index, staleness_max_days)
        missing = stale.astype(float)
        return norm, missing
    except Exception as exc:
        logger.error("ml_feature_error feature=rates_level error=%s", exc)
        return (
            _default_series(index, 0.5),
            _default_series(index, 1.0),
        )
