"""
NovaCycle Technical Indicators
================================
All formulas are commented with mathematical notation.

Rules enforced throughout:
  - SMA50/SMA200, MACD, ADX, VIX regime MUST NOT use extended-hours candles.
  - Time-decay weighting applied via apply_time_decay().
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# New indicator configuration (VOO-only, in-memory calculations)
ADAPTIVE_RSI_PERIODS = {
    "macro_shock": 7,
    "trending": 10,
    "compressed": 21,
    "calm": 14,
}
EMA_RIBBON_PERIODS = [8, 13, 21, 34, 55]
ATR_COMPRESSION_LOOKBACK = 100
BOLLINGER_SLOPE_WINDOW = 10
TREND_STRENGTH_ATR_WEIGHT = 0.25
TREND_STRENGTH_RSI_WEIGHT = 0.25
TREND_STRENGTH_EMA_WEIGHT = 0.50


class TechnicalIndicators:
    """Compute all technical indicators needed by the signal engine."""

    # ──────────────────────────────────────────────────────────────────────────
    # RSI
    # ──────────────────────────────────────────────────────────────────────────

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (Wilder smoothing).

        Formula:
          delta   = Close[t] - Close[t-1]
          gain    = delta if delta > 0 else 0
          loss    = |delta| if delta < 0 else 0

          avg_gain = EWM(gain, alpha = 1/period, adjust=False)
          avg_loss = EWM(loss, alpha = 1/period, adjust=False)

          RS  = avg_gain / avg_loss
          RSI = 100 - (100 / (1 + RS))

        Returns:
            pd.Series, range [0, 100]
        """
        try:
            delta = prices.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)

            # Wilder's smoothing = EWM with alpha = 1/period
            alpha = 1.0 / period
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi.fillna(50.0)
        except Exception as exc:
            logger.error("compute_rsi error: %s", exc)
            return pd.Series(50.0, index=prices.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Stochastic Oscillator
    # ──────────────────────────────────────────────────────────────────────────

    def compute_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> dict:
        """
        Stochastic Oscillator.

        Formula:
          Lowest_Low   = min(Low[-k_period:])
          Highest_High = max(High[-k_period:])

          %K = (Close - Lowest_Low) / (Highest_High - Lowest_Low) * 100
          %D = SMA(%K, d_period)

        Returns:
            {"k": pd.Series [0-100], "d": pd.Series [0-100]}
        """
        try:
            lowest_low = low.rolling(k_period).min()
            highest_high = high.rolling(k_period).max()
            denom = (highest_high - lowest_low).replace(0, np.nan)

            k = (close - lowest_low) / denom * 100.0
            k = k.fillna(50.0)
            d = k.rolling(d_period).mean().fillna(50.0)
            return {"k": k, "d": d}
        except Exception as exc:
            logger.error("compute_stochastic error: %s", exc)
            n = len(close)
            return {
                "k": pd.Series(50.0, index=close.index),
                "d": pd.Series(50.0, index=close.index),
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Stochastic RSI
    # ──────────────────────────────────────────────────────────────────────────

    def compute_stoch_rsi(
        self,
        prices: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> dict:
        """
        Stochastic RSI.

        Formula:
          RSI  = compute_rsi(prices, period)

          StochRSI = (RSI - min(RSI, period)) / (max(RSI, period) - min(RSI, period))

          StochRSI_K = SMA(StochRSI, smooth_k)  × 100
          StochRSI_D = SMA(StochRSI_K, smooth_d)

        Returns:
            {"k": pd.Series [0-100], "d": pd.Series [0-100]}
        """
        try:
            rsi = self.compute_rsi(prices, period)
            rsi_min = rsi.rolling(period).min()
            rsi_max = rsi.rolling(period).max()
            denom = (rsi_max - rsi_min).replace(0, np.nan)

            stoch_rsi = (rsi - rsi_min) / denom
            stoch_rsi = stoch_rsi.fillna(0.5)

            k = stoch_rsi.rolling(smooth_k).mean().fillna(0.5) * 100.0
            d = k.rolling(smooth_d).mean().fillna(50.0)
            return {"k": k, "d": d}
        except Exception as exc:
            logger.error("compute_stoch_rsi error: %s", exc)
            return {
                "k": pd.Series(50.0, index=prices.index),
                "d": pd.Series(50.0, index=prices.index),
            }

    # ──────────────────────────────────────────────────────────────────────────
    # MACD
    # ──────────────────────────────────────────────────────────────────────────

    def compute_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict:
        """
        Moving Average Convergence/Divergence.

        Formula:
          EMA_fast      = EMA(Close, fast)      [k = 2/(fast+1)]
          EMA_slow      = EMA(Close, slow)      [k = 2/(slow+1)]

          MACD_line     = EMA_fast - EMA_slow
          Signal_line   = EMA(MACD_line, signal)
          Histogram     = MACD_line - Signal_line

        Returns:
            {"macd": pd.Series, "signal": pd.Series, "histogram": pd.Series}
        """
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return {"macd": macd_line, "signal": signal_line, "histogram": histogram}
        except Exception as exc:
            logger.error("compute_macd error: %s", exc)
            zeros = pd.Series(0.0, index=prices.index)
            return {"macd": zeros, "signal": zeros, "histogram": zeros}

    # ──────────────────────────────────────────────────────────────────────────
    # SMA / EMA
    # ──────────────────────────────────────────────────────────────────────────

    def compute_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.

        Formula:
          SMA(t) = Σ prices[t-period+1 .. t] / period
        """
        try:
            return prices.rolling(period).mean()
        except Exception as exc:
            logger.error("compute_sma error: %s", exc)
            return pd.Series(np.nan, index=prices.index)

    def compute_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.

        Formula:
          k       = 2 / (period + 1)
          EMA(t)  = Close(t) × k + EMA(t-1) × (1 - k)
        """
        try:
            return prices.ewm(span=period, adjust=False).mean()
        except Exception as exc:
            logger.error("compute_ema error: %s", exc)
            return pd.Series(np.nan, index=prices.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Bollinger Bands
    # ──────────────────────────────────────────────────────────────────────────

    def compute_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> dict:
        """
        Bollinger Bands.

        Formula:
          Middle  = SMA(Close, period)
          Std     = StdDev(Close, period)

          Upper   = Middle + std_dev × Std
          Lower   = Middle - std_dev × Std

          %B      = (Close - Lower) / (Upper - Lower)
          Width   = (Upper - Lower) / Middle          [bandwidth]

        Returns:
            {
              "upper": pd.Series, "middle": pd.Series, "lower": pd.Series,
              "pct_b": pd.Series,  "bandwidth": pd.Series
            }
        """
        try:
            middle = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = middle + std_dev * std
            lower = middle - std_dev * std
            band_range = (upper - lower).replace(0, np.nan)
            pct_b = (prices - lower) / band_range
            bandwidth = band_range / middle.replace(0, np.nan)
            return {
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "pct_b": pct_b.fillna(0.5),
                "bandwidth": bandwidth.fillna(0.0),
            }
        except Exception as exc:
            logger.error("compute_bollinger_bands error: %s", exc)
            n = len(prices)
            return {
                "upper": prices,
                "middle": prices,
                "lower": prices,
                "pct_b": pd.Series(0.5, index=prices.index),
                "bandwidth": pd.Series(0.0, index=prices.index),
            }

    # ──────────────────────────────────────────────────────────────────────────
    # CCI
    # ──────────────────────────────────────────────────────────────────────────

    def compute_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """
        Commodity Channel Index.

        Formula:
          TP      = (High + Low + Close) / 3
          SMA_TP  = SMA(TP, period)
          MAD     = mean( |TP[i] - SMA_TP| ) over period  [Mean Absolute Deviation]
          CCI     = (TP - SMA_TP) / (0.015 × MAD)

        Returns:
            pd.Series (typical range: -200 to +200)
        """
        try:
            tp = (high + low + close) / 3.0
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
            return cci.fillna(0.0)
        except Exception as exc:
            logger.error("compute_cci error: %s", exc)
            return pd.Series(0.0, index=close.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Williams %R
    # ──────────────────────────────────────────────────────────────────────────

    def compute_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Williams %R.

        Formula:
          Highest_High = max(High[-period:])
          Lowest_Low   = min(Low[-period:])

          %R = (Highest_High - Close) / (Highest_High - Lowest_Low) × -100

        Returns:
            pd.Series, range [-100, 0]
        """
        try:
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            denom = (highest_high - lowest_low).replace(0, np.nan)
            wr = (highest_high - close) / denom * -100.0
            return wr.fillna(-50.0)
        except Exception as exc:
            logger.error("compute_williams_r error: %s", exc)
            return pd.Series(-50.0, index=close.index)

    # ──────────────────────────────────────────────────────────────────────────
    # ATR
    # ──────────────────────────────────────────────────────────────────────────

    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average True Range (Wilder smoothing).

        Formula:
          TR(t)  = max(
                     High(t) - Low(t),
                     |High(t) - Close(t-1)|,
                     |Low(t)  - Close(t-1)|
                   )
          ATR    = EWM(TR, alpha = 1/period, adjust=False)

        Returns:
            pd.Series (in price units)
        """
        try:
            prev_close = close.shift(1)
            tr = pd.concat(
                [
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)

            alpha = 1.0 / period
            atr = tr.ewm(alpha=alpha, adjust=False).mean()
            return atr
        except Exception as exc:
            logger.error("compute_atr error: %s", exc)
            return pd.Series(0.0, index=close.index)

    # ──────────────────────────────────────────────────────────────────────────
    # ADX
    # ──────────────────────────────────────────────────────────────────────────

    def compute_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average Directional Index (Wilder smoothing).

        Formula:
          +DM(t) = High(t) - High(t-1)  if > 0 and > PrevLow - Low  else 0
          -DM(t) = Low(t-1) - Low(t)    if > 0 and > High - PrevHigh else 0

          TR     = max(High-Low, |High-PrevClose|, |Low-PrevClose|)

          Smoothed +DM = EWM(+DM, alpha=1/period)
          Smoothed -DM = EWM(-DM, alpha=1/period)
          ATR          = EWM(TR,  alpha=1/period)

          +DI = 100 × Smoothed+DM / ATR
          -DI = 100 × Smoothed-DM / ATR

          DX  = 100 × |+DI - -DI| / (+DI + -DI)
          ADX = EWM(DX, alpha=1/period)

        Returns:
            pd.Series (range 0–100; > 25 = trending)
        """
        try:
            alpha = 1.0 / period

            prev_high = high.shift(1)
            prev_low = low.shift(1)
            prev_close = close.shift(1)

            up_move = high - prev_high
            down_move = prev_low - low

            plus_dm = np.where((up_move > 0) & (up_move > down_move), up_move, 0.0)
            minus_dm = np.where((down_move > 0) & (down_move > up_move), down_move, 0.0)

            plus_dm_s = pd.Series(plus_dm, index=high.index)
            minus_dm_s = pd.Series(minus_dm, index=high.index)

            tr = pd.concat(
                [
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)

            atr = tr.ewm(alpha=alpha, adjust=False).mean()
            s_plus_dm = plus_dm_s.ewm(alpha=alpha, adjust=False).mean()
            s_minus_dm = minus_dm_s.ewm(alpha=alpha, adjust=False).mean()

            atr_safe = atr.replace(0, np.nan)
            plus_di = 100.0 * s_plus_dm / atr_safe
            minus_di = 100.0 * s_minus_dm / atr_safe

            di_sum = (plus_di + minus_di).replace(0, np.nan)
            dx = 100.0 * (plus_di - minus_di).abs() / di_sum

            adx = dx.ewm(alpha=alpha, adjust=False).mean()
            return adx.fillna(0.0)
        except Exception as exc:
            logger.error("compute_adx error: %s", exc)
            return pd.Series(0.0, index=close.index)

    # ──────────────────────────────────────────────────────────────────────────
    # VIX Regime
    # ──────────────────────────────────────────────────────────────────────────

    def compute_vix_regime(self, vix_values: pd.Series) -> pd.Series:
        """
        Classify VIX into volatility regimes.

        Thresholds:
          VIX < 15       → 'LOW'
          15 ≤ VIX < 25  → 'NORMAL'
          25 ≤ VIX < 35  → 'HIGH'
          VIX ≥ 35       → 'EXTREME'

        Returns:
            pd.Series of str ('LOW', 'NORMAL', 'HIGH', 'EXTREME')
        """
        try:
            def classify(v: float) -> str:
                if v < 15:
                    return "LOW"
                elif v < 25:
                    return "NORMAL"
                elif v < 35:
                    return "HIGH"
                else:
                    return "EXTREME"

            return vix_values.apply(classify)
        except Exception as exc:
            logger.error("compute_vix_regime error: %s", exc)
            return pd.Series("NORMAL", index=vix_values.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Volatility regime helper (for adaptive RSI and trend strength)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_volatility_regime(
        self,
        close: pd.Series,
        atr: pd.Series | None = None,
        vix_regime: pd.Series | None = None,
    ) -> pd.Series:
        """
        Classify volatility into one of four regimes used by the adaptive
        RSI and the trend-strength index. Reuses the existing ATR and VIX
        regime series to avoid recomputing volatility primitives.

        Regimes:
          macro_shock : rolling return std dev explodes (>2x median) or
                        VIX regime is EXTREME
          compressed  : volatility collapses (<0.5x median) or ATR is tiny
                        relative to price (<0.1%)
          trending    : ATR-normalised volatility is expanding
          calm        : none of the above
        """
        try:
            ret_std = close.pct_change().rolling(20, min_periods=5).std()
            baseline = ret_std.rolling(100, min_periods=20).median()
            ratio = (ret_std / baseline.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(1.0)

            atr_norm = pd.Series(0.0, index=close.index)
            if atr is not None and not atr.empty:
                atr_norm = (
                    (atr.reindex(close.index) / close.replace(0, np.nan))
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

            macro = ratio > 2.0
            if vix_regime is not None and not vix_regime.empty:
                vix_aligned = vix_regime.reindex(close.index, method="ffill").fillna(
                    "NORMAL"
                )
                macro = macro | (vix_aligned == "EXTREME")

            compressed = (ratio < 0.5) | (atr_norm < 0.001)
            trending = atr_norm > atr_norm.rolling(20, min_periods=5).mean()

            regimes = np.select(
                [macro, compressed, trending],
                ["macro_shock", "compressed", "trending"],
                default="calm",
            )
            return pd.Series(regimes, index=close.index)
        except Exception as exc:
            logger.error("compute_volatility_regime error: %s", exc)
            return pd.Series("calm", index=close.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Adaptive RSI
    # ──────────────────────────────────────────────────────────────────────────

    def compute_adaptive_rsi(
        self,
        prices: pd.Series,
        volatility_regime: pd.Series | None = None,
        base_period: int = 14,
    ) -> pd.Series:
        """
        RSI whose lookback period adapts to the current volatility regime.

        Period mapping:
          macro_shock → 7  (very sensitive)
          trending    → 10
          compressed  → 21 (smoother, range-bound)
          calm        → base_period (14)

        Falls back to the standard RSI on any error.
        """
        try:
            if volatility_regime is None or volatility_regime.empty:
                return self.compute_rsi(prices, base_period)

            # Precompute only the distinct periods needed by the regime map.
            rsi_by_period = {
                period: self.compute_rsi(prices, period)
                for period in set(ADAPTIVE_RSI_PERIODS.values())
            }

            regime = (
                volatility_regime.reindex(prices.index, method="ffill")
                .fillna("calm")
            )
            default_period = ADAPTIVE_RSI_PERIODS.get("calm", base_period)

            adaptive = np.select(
                [
                    regime == "macro_shock",
                    regime == "trending",
                    regime == "compressed",
                    regime == "calm",
                ],
                [
                    rsi_by_period[ADAPTIVE_RSI_PERIODS["macro_shock"]],
                    rsi_by_period[ADAPTIVE_RSI_PERIODS["trending"]],
                    rsi_by_period[ADAPTIVE_RSI_PERIODS["compressed"]],
                    rsi_by_period[default_period],
                ],
                default=rsi_by_period[default_period],
            )
            return pd.Series(adaptive, index=prices.index).fillna(50.0)
        except Exception as exc:
            logger.error("compute_adaptive_rsi error: %s", exc)
            return self.compute_rsi(prices, base_period)

    # ──────────────────────────────────────────────────────────────────────────
    # EMA Ribbon Alignment
    # ──────────────────────────────────────────────────────────────────────────

    def compute_ema_ribbon(self, prices: pd.Series) -> dict:
        """
        EMA ribbon (8, 13, 21, 34, 55) and its alignment state.

        Alignment:
          bullish : shorter EMAs above longer EMAs (8 > 13 > 21 > 34 > 55)
          bearish : shorter EMAs below longer EMAs (8 < 13 < 21 < 34 < 55)
          neutral : mixed / incomplete
        """
        try:
            emas = {
                f"ema{p}": prices.ewm(span=p, adjust=False).mean()
                for p in EMA_RIBBON_PERIODS
            }
            e8, e13, e21, e34, e55 = (
                emas["ema8"],
                emas["ema13"],
                emas["ema21"],
                emas["ema34"],
                emas["ema55"],
            )

            bullish = (e8 > e13) & (e13 > e21) & (e21 > e34) & (e34 > e55)
            bearish = (e8 < e13) & (e13 < e21) & (e21 < e34) & (e34 < e55)

            alignment = np.select(
                [bullish, bearish],
                ["bullish", "bearish"],
                default="neutral",
            )
            emas["alignment"] = pd.Series(alignment, index=prices.index)
            return emas
        except Exception as exc:
            logger.error("compute_ema_ribbon error: %s", exc)
            emas = {
                f"ema{p}": pd.Series(np.nan, index=prices.index)
                for p in EMA_RIBBON_PERIODS
            }
            emas["alignment"] = pd.Series("neutral", index=prices.index)
            return emas

    # ──────────────────────────────────────────────────────────────────────────
    # ATR Compression
    # ──────────────────────────────────────────────────────────────────────────

    def compute_atr_compression_score(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        lookback: int = ATR_COMPRESSION_LOOKBACK,
    ) -> pd.Series:
        """
        ATR compression score in [0, 1].

        Detects low-volatility squeezes by comparing the current ATR-normalised
        range to its recent rolling median. A score near 1 means the current
        volatility is far below its baseline (a squeeze); a score near 0 means
        volatility is at or above baseline.
        """
        try:
            atr = self.compute_atr(high, low, close, period)
            atr_norm = (
                (atr / close.replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            median_atr_norm = atr_norm.rolling(lookback, min_periods=period).median()
            score = 1.0 - (atr_norm / median_atr_norm.replace(0, np.nan))
            return score.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        except Exception as exc:
            logger.error("compute_atr_compression_score error: %s", exc)
            return pd.Series(0.0, index=close.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Bollinger Band Slope
    # ──────────────────────────────────────────────────────────────────────────

    def compute_bollinger_slope(
        self,
        prices: pd.Series,
        bollinger_data: dict | None = None,
        window: int = BOLLINGER_SLOPE_WINDOW,
    ) -> pd.Series:
        """
        Slope of the Bollinger middle band (SMA20) normalised by its own level.

        Reuses existing Bollinger band values when they are passed in; otherwise
        recomputes the middle band only. Positive values indicate upward
        directional pressure; negative values indicate downward pressure.
        """
        try:
            if bollinger_data is not None and "middle" in bollinger_data:
                middle = bollinger_data["middle"]
            else:
                middle = prices.rolling(20, min_periods=1).mean()
            middle = middle.reindex(prices.index)
            slope = middle.diff(window) / middle.shift(window).replace(0, np.nan)
            return slope.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception as exc:
            logger.error("compute_bollinger_slope error: %s", exc)
            return pd.Series(0.0, index=prices.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Trend-Strength Index
    # ──────────────────────────────────────────────────────────────────────────

    def compute_trend_strength_index(
        self,
        ema_alignment: pd.Series,
        rsi: pd.Series,
        atr_compression_score: pd.Series,
    ) -> pd.Series:
        """
        Lightweight trend-strength score in [0, 1] combining EMA ribbon
        alignment, RSI directional pressure, and ATR compression.

        Components:
          EMA alignment        : bullish=+1, bearish=-1, neutral=0
          RSI pressure         : (RSI - 50) / 50, clipped to [-1, 1]
          ATR compression      : 1 - 2*score (high compression → weak trend)

        Weights: EMA 50%, RSI 25%, ATR 25%. The combined score is shifted to
        [0, 1].
        """
        try:
            ema_score = (
                ema_alignment.map({"bullish": 1.0, "bearish": -1.0, "neutral": 0.0})
                .fillna(0.0)
            )
            rsi_score = ((rsi - 50.0) / 50.0).fillna(0.0).clip(-1.0, 1.0)
            atr_score = (1.0 - 2.0 * atr_compression_score).clip(-1.0, 1.0)

            combined = (
                TREND_STRENGTH_EMA_WEIGHT * ema_score
                + TREND_STRENGTH_RSI_WEIGHT * rsi_score
                + TREND_STRENGTH_ATR_WEIGHT * atr_score
            )
            score = (combined + 1.0) / 2.0
            return score.clip(0.0, 1.0)
        except Exception as exc:
            logger.error("compute_trend_strength_index error: %s", exc)
            return pd.Series(0.5, index=ema_alignment.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Time-decay weighting
    # ──────────────────────────────────────────────────────────────────────────

    def apply_time_decay(
        self,
        values: pd.Series,
        lambda_val: float,
        time_unit: str = "days",
    ) -> pd.Series:
        """
        Apply exponential time-decay weights to a series.

        Formula (regular hours):
          Weight(t) = 1.0 × exp(-lambda × age)

        Formula (extended hours, treated as short-term):
          Weight(t) = 0.5 × exp(-lambda_short × age_in_minutes)

        Args:
            values:     Series indexed by pd.Timestamp
            lambda_val: decay constant (LAMBDA_LONG for daily, LAMBDA_SHORT for 5-min)
            time_unit:  'days' or 'minutes'

        Returns:
            pd.Series of weights in (0, 1]
        """
        try:
            now = pd.Timestamp.utcnow().tz_localize(None)
            if time_unit == "days":
                ages = (now - values.index.tz_localize(None)).days
            else:  # minutes
                ages = (now - values.index.tz_localize(None)).total_seconds() / 60.0

            weights = pd.Series(
                [math.exp(-lambda_val * max(0.0, a)) for a in ages],
                index=values.index,
            )
            return weights
        except Exception as exc:
            logger.error("apply_time_decay error: %s", exc)
            return pd.Series(1.0, index=values.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Compute all indicators
    # ──────────────────────────────────────────────────────────────────────────

    def compute_all(
        self,
        df: pd.DataFrame,
        vix_df: pd.DataFrame,
        exclude_extended: bool = True,
    ) -> dict:
        """
        Compute the full indicator set from VOO + VIX data.

        Rules:
          - SMA50, SMA200, MACD, ADX, VIX regime MUST NEVER use
            extended-hours candles (is_extended_hours=True).
          - Short-term indicators use all candles.

        Args:
            df:               VOO candle DataFrame (must have OHLCV + is_extended_hours)
            vix_df:           VIX daily DataFrame
            exclude_extended: Filter extended-hours before computing long-term indicators

        Returns:
            dict with all indicator values (scalar or pd.Series)
        """
        result: dict = {}

        if df.empty:
            logger.warning("compute_all: empty DataFrame supplied")
            return result

        try:
            # Overlapping vendor fetches can briefly produce duplicate
            # timestamps in an in-memory frame even though the DB uniqueness
            # checks prevent duplicate daily rows. Reindexing a duplicate
            # index raises and silently removes the macro feature set.
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if not vix_df.empty:
                vix_df = vix_df.sort_index()
                vix_df = vix_df[~vix_df.index.duplicated(keep="last")]
            # ── Regular-hours-only slice (for long-term indicators) ────────────
            if exclude_extended and "is_extended_hours" in df.columns:
                reg_df = df[df["is_extended_hours"] == False].copy()
            else:
                reg_df = df.copy()

            close = df["close"]
            high = df["high"]
            low = df["low"]
            reg_close = reg_df["close"]
            reg_high = reg_df["high"]
            reg_low = reg_df["low"]

            # ── Long-term indicators (regular hours only) ──────────────────────
            result["sma50"] = self.compute_sma(reg_close, 50)
            result["sma200"] = self.compute_sma(reg_close, 200)
            result["sma20"] = self.compute_sma(reg_close, 20)

            macd_data = self.compute_macd(reg_close)
            result["macd"] = macd_data["macd"]
            result["macd_signal"] = macd_data["signal"]
            result["macd_histogram"] = macd_data["histogram"]

            result["adx"] = self.compute_adx(reg_high, reg_low, reg_close)
            result["atr"] = self.compute_atr(reg_high, reg_low, reg_close)

            # ── VIX regime ─────────────────────────────────────────────────────
            if not vix_df.empty and "close" in vix_df.columns:
                vix_close = vix_df["close"].astype(float).sort_index()
                vix_regime = self.compute_vix_regime(vix_close)
                # VIX is a daily series. Align only from the latest known
                # completed VIX observation; never backfill from the future.
                vix_aligned = vix_close.reindex(reg_df.index, method="ffill")
                regime_aligned = vix_regime.reindex(reg_df.index, method="ffill")
                result["vix_regime"] = regime_aligned
                result["vix_level"] = vix_aligned
                result["vix_change_5d"] = vix_aligned.pct_change(5).replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0.0)
                result["vix_percentile_1y"] = vix_aligned.rolling(
                    min(252, max(20, len(vix_close))), min_periods=20
                ).rank(pct=True).fillna(0.5)
                result["vix_missing"] = vix_aligned.isna()
                result["vix_latest"] = float(vix_close.iloc[-1])
                result["vix_regime_latest"] = (
                    vix_regime.iloc[-1]
                )
            else:
                result["vix_regime"] = pd.Series("NORMAL", index=reg_df.index)
                result["vix_level"] = pd.Series(np.nan, index=reg_df.index)
                result["vix_change_5d"] = pd.Series(0.0, index=reg_df.index)
                result["vix_percentile_1y"] = pd.Series(0.5, index=reg_df.index)
                result["vix_missing"] = pd.Series(True, index=reg_df.index)
                result["vix_latest"] = 20.0
                result["vix_regime_latest"] = "NORMAL"

            # ── Short-term indicators (all candles) ────────────────────────────
            result["rsi"] = self.compute_rsi(close)
            result["stoch"] = self.compute_stochastic(high, low, close)
            result["stoch_rsi"] = self.compute_stoch_rsi(close)
            result["bollinger"] = self.compute_bollinger_bands(close)
            result["cci"] = self.compute_cci(high, low, close)
            result["williams_r"] = self.compute_williams_r(high, low, close)
            result["atr_all"] = self.compute_atr(high, low, close)

            # ── VOO-only upgraded indicators (in-memory, backward-compatible) ──────
            volatility_regime = self._compute_volatility_regime(
                close,
                atr=result["atr"].reindex(close.index, method="ffill"),
                vix_regime=result["vix_regime"].reindex(close.index, method="ffill"),
            )
            result["adaptive_rsi"] = self.compute_adaptive_rsi(close, volatility_regime)
            result["ema_ribbon"] = self.compute_ema_ribbon(close)
            result["atr_compression_score"] = self.compute_atr_compression_score(
                high, low, close
            )
            result["bollinger_slope"] = self.compute_bollinger_slope(
                close, result["bollinger"]
            )
            result["trend_strength_index"] = self.compute_trend_strength_index(
                result["ema_ribbon"]["alignment"],
                result["rsi"],
                result["atr_compression_score"],
            )

            # ── Scalars (latest values) ────────────────────────────────────────
            def _last(series: pd.Series) -> Optional[float]:
                if series.empty or series.isna().all():
                    return None
                return float(series.dropna().iloc[-1])

            def _last_label(series: pd.Series) -> Optional[str]:
                if series.empty or series.isna().all():
                    return None
                val = series.dropna().iloc[-1]
                return str(val) if val is not None else None

            result["latest"] = {
                "close": _last(close),
                "sma50": _last(result["sma50"]),
                "sma200": _last(result["sma200"]),
                "sma20": _last(result["sma20"]),
                "macd": _last(result["macd"]),
                "macd_signal": _last(result["macd_signal"]),
                "macd_histogram": _last(result["macd_histogram"]),
                "adx": _last(result["adx"]),
                "atr": _last(result["atr"]),
                "rsi": _last(result["rsi"]),
                "stoch_k": _last(result["stoch"]["k"]),
                "stoch_d": _last(result["stoch"]["d"]),
                "stoch_rsi_k": _last(result["stoch_rsi"]["k"]),
                "stoch_rsi_d": _last(result["stoch_rsi"]["d"]),
                "bb_upper": _last(result["bollinger"]["upper"]),
                "bb_middle": _last(result["bollinger"]["middle"]),
                "bb_lower": _last(result["bollinger"]["lower"]),
                "bb_pct_b": _last(result["bollinger"]["pct_b"]),
                "bb_bandwidth": _last(result["bollinger"]["bandwidth"]),
                "cci": _last(result["cci"]),
                "williams_r": _last(result["williams_r"]),
                "vix": result.get("vix_latest"),
                "vix_regime": result.get("vix_regime_latest"),
                # New VOO-only upgraded indicators
                "adaptive_rsi": _last(result["adaptive_rsi"]),
                "ema_ribbon_alignment": _last_label(result["ema_ribbon"]["alignment"]),
                "atr_compression_score": _last(result["atr_compression_score"]),
                "bollinger_slope": _last(result["bollinger_slope"]),
                "trend_strength_index": _last(result["trend_strength_index"]),
            }

            # ── Return metrics ─────────────────────────────────────────────────
            if len(reg_close) >= 5:
                result["return_5d"] = float(
                    (reg_close.iloc[-1] - reg_close.iloc[-5]) / reg_close.iloc[-5]
                )
            else:
                result["return_5d"] = 0.0

            if len(reg_close) >= 10:
                result["return_10d"] = float(
                    (reg_close.iloc[-1] - reg_close.iloc[-10]) / reg_close.iloc[-10]
                )
            else:
                result["return_10d"] = 0.0

            if len(reg_close) >= 20:
                result["return_20d"] = float(
                    (reg_close.iloc[-1] - reg_close.iloc[-20]) / reg_close.iloc[-20]
                )
            else:
                result["return_20d"] = 0.0

            # ── SMA distance ───────────────────────────────────────────────────
            latest_close = _last(reg_close)
            sma20_val = _last(result["sma20"])
            if latest_close and sma20_val and sma20_val != 0:
                result["sma20_distance"] = (latest_close - sma20_val) / sma20_val
            else:
                result["sma20_distance"] = 0.0

        except Exception as exc:
            logger.error("compute_all error: %s", exc)

        return result
