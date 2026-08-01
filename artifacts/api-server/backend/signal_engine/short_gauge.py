"""
NovaCycle Short-Trend Gauge
============================
Aggregates short-term (5-min) technical indicators, extended-hours data,
gap analysis, liquidity, and ML output into a score in [-100, +100].

Score composition:
  indicator_score  (individual caps: ±8 regular / ±4 extended)
  ml_score         = calibrated probability relative to its neutral base rate
                     → maps [0,1] to [-40,+40]
  total_score      = (indicator_score + ml_score) × time_decay_weight

BUY  threshold: total_score > +50
SELL threshold: total_score < -50
NEUTRAL:        otherwise

Extended-hours calibration:
  Indicator contributions are already halved for extended-hours bars
  (ext_mult = 0.5 inside compute_indicator_score). The time-decay
  base_weight is therefore set to 1.0 for ALL sessions — applying a
  second 0.5 multiplier would make the threshold mathematically
  unreachable during extended hours and would suppress strong setups
  that only appear pre/post-market. The indicator halving alone is
  sufficient to reflect the lower reliability of thin-session data.

  Extended max raw score:  13 (indicators) + 40 (ML) = 53  → clears ±50
  Regular   max raw score: 26 (indicators) + 40 (ML) = 66  → clears ±50

Liquidity filter:
  LiquidityScore = Volume_extended / Volume_regular
  If < 0.15: all indicator weights × 0.5

Gap influence (applied AFTER liquidity filter):
  gap_up   → subtract 10 from the bullish (buy) components
  gap_down → subtract 10 from the bearish (sell) components

Time-decay:
  base_weight = 1.0 (same for regular and extended sessions)
  Weight(t)   = base_weight × exp(−LAMBDA_SHORT × age_in_minutes)
"""

import logging
import math

from config import settings

logger = logging.getLogger(__name__)

# ML score mapping: [0,1] → [-40, +40]. The neutral probability is normally
# the short label's calibrated OOS positive rate, not necessarily 0.5.
_ML_WEIGHT = 80.0
_ML_OFFSET = 40.0
_SCORE_MIN = -100.0
_SCORE_MAX = 100.0


class ShortTrendGauge:
    """Compute short-trend gauge score from 5-min indicators + ML."""

    # ──────────────────────────────────────────────────────────────────────────
    # Indicator scoring
    # ──────────────────────────────────────────────────────────────────────────

    def compute_indicator_score(
        self,
        indicators: dict,
        is_extended: bool,
        liquidity_score: float,
        gap_type: str,
        gap_momentum: float | None = None,
        neutral_probability: float = 0.5,
    ) -> tuple[float, dict]:
        """
        Compute the raw indicator score for the short-trend gauge.

        Indicator contributions:
          RSI:
            < 30  oversold  → +8 regular / +4 extended
            > 70  overbought → -8 regular / -4 extended

          StochRSI_K:
            < 20  → +6 regular / +3 extended
            > 80  → -6 regular / -3 extended

          Stochastic %K:
            < 20  → +4 regular / +2 extended
            > 80  → -4 regular / -2 extended

          Bollinger %B:
            < 0   (below lower band) → +8 regular / +4 extended
            > 1   (above upper band) → -8 regular / -4 extended

        Liquidity filter:
          if liquidity_score < 0.15: all contributions × 0.5

        Gap influence (applied after liquidity):
          gap_up   → subtract 10 from positive (bullish) total
          gap_down → subtract 10 from negative (bearish) total

        Returns:
            (total_indicator_score: float, breakdown: dict)
        """
        breakdown: dict = {}
        total = 0.0

        # Multiplier: regular=1.0, extended=0.5
        ext_mult = 0.5 if is_extended else 1.0

        latest = indicators.get("latest", {})

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = latest.get("rsi", 50.0) or 50.0
        if rsi < 30.0:
            rsi_score = (8.0 if not is_extended else 4.0)
        elif rsi > 70.0:
            rsi_score = -(8.0 if not is_extended else 4.0)
        else:
            rsi_score = 0.0
        breakdown["rsi"] = rsi_score
        total += rsi_score

        # ── StochRSI K ────────────────────────────────────────────────────────
        srk = latest.get("stoch_rsi_k", 50.0) or 50.0
        if srk < 20.0:
            srk_score = (6.0 if not is_extended else 3.0)
        elif srk > 80.0:
            srk_score = -(6.0 if not is_extended else 3.0)
        else:
            srk_score = 0.0
        breakdown["stoch_rsi_k"] = srk_score
        total += srk_score

        # ── Stochastic %K ─────────────────────────────────────────────────────
        stoch_k = latest.get("stoch_k", 50.0) or 50.0
        if stoch_k < 20.0:
            stk_score = (4.0 if not is_extended else 2.0)
        elif stoch_k > 80.0:
            stk_score = -(4.0 if not is_extended else 2.0)
        else:
            stk_score = 0.0
        breakdown["stoch_k"] = stk_score
        total += stk_score

        # ── Bollinger %B ──────────────────────────────────────────────────────
        bb_pb = latest.get("bb_pct_b", 0.5) or 0.5
        if bb_pb < 0.0:
            bb_score = (8.0 if not is_extended else 4.0)
        elif bb_pb > 1.0:
            bb_score = -(8.0 if not is_extended else 4.0)
        else:
            bb_score = 0.0
        breakdown["bollinger_pct_b"] = bb_score
        total += bb_score

        # ── Liquidity filter: if < 0.15, weights × 0.5 ────────────────────────
        liquidity_adjusted = False
        if liquidity_score < settings.LIQUIDITY_SCORE_THRESHOLD:
            total *= 0.5
            for k in breakdown:
                breakdown[k] *= 0.5
            breakdown["liquidity_adjustment"] = f"× 0.5 (score={liquidity_score:.4f})"
            liquidity_adjusted = True

        # ── Gap influence ──────────────────────────────────────────────────────
        gap_adjustment = 0.0
        g = (gap_type or "none").lower()
        if g == "gap_up":
            # Gap-up = likely overbought open → penalise BUY (bullish) components
            # Reduce positive total by 10 (can push toward neutral/bearish)
            gap_adjustment = -10.0
            total = max(total + gap_adjustment, total - 10.0)
            breakdown["gap_influence"] = f"gap_up: −10 on buy score"
        elif g == "gap_down":
            # Gap-down = likely oversold open → penalise SELL (bearish) components
            gap_adjustment = +10.0
            total = min(total + gap_adjustment, total + 10.0)
            breakdown["gap_influence"] = f"gap_down: −10 on sell score"

        # ── Gap follow-through (momentum) influence ───────────────────────────
        # Additive layer on top of gap influence. gap_momentum is the signed %
        # move over the first 30 minutes of the regular session relative to
        # the gap direction (see DataFetcher.compute_gap_momentum):
        #   momentum > +threshold → gap is following through: boost the score
        #     toward the gap direction (gap_up → more bullish, gap_down →
        #     more bearish).
        #   momentum < -threshold → gap is fading: push the score away from
        #     the gap direction (downgrades signals that chase a fading gap).
        # None (no gap / not enough post-open data) → no effect.
        if gap_momentum is not None and g in ("gap_up", "gap_down"):
            direction = 1.0 if g == "gap_up" else -1.0
            boost = settings.GAP_MOMENTUM_SCORE_BOOST
            if gap_momentum >= settings.GAP_MOMENTUM_THRESHOLD:
                adj = direction * boost
                total += adj
                breakdown["gap_momentum_influence"] = (
                    f"follow-through ({gap_momentum:+.2f}%): {adj:+.1f}"
                )
            elif gap_momentum <= -settings.GAP_MOMENTUM_THRESHOLD:
                adj = -direction * boost
                total += adj
                breakdown["gap_momentum_influence"] = (
                    f"fading ({gap_momentum:+.2f}%): {adj:+.1f}"
                )

        return total, breakdown

    # ──────────────────────────────────────────────────────────────────────────
    # Full score computation
    # ──────────────────────────────────────────────────────────────────────────

    def compute_score(
        self,
        indicators: dict,
        ml_prediction: float,
        is_extended: bool,
        liquidity_score: float,
        gap_type: str,
        age_in_minutes: float = 0.0,
        gap_momentum: float | None = None,
        neutral_probability: float = 0.5,
    ) -> dict:
        """
        Compute the final short-trend gauge score.

        Formula:
          indicator_score = Σ indicator contributions (with liquidity + gap)
          ml_score        = ml_prediction centered on neutral_probability,
                          bounded to [-40, +40]
          raw_score       = indicator_score + ml_score

          # Time-decay
          base_weight = 0.5 if is_extended else 1.0
          Weight(t)   = base_weight × exp(−LAMBDA_SHORT × age_in_minutes)
          total_score = clamp(raw_score × Weight(t), −100, +100)

        Signal rules:
          score >  SHORT_BUY_THRESHOLD  (+50) → 'buy'
          score <  SHORT_SELL_THRESHOLD (−50) → 'sell'
          otherwise                           → 'neutral'

        Liquidity suppression (applied here too):
          If liquidity_score < 0.15, thresholds are raised by 25%:
            effective_buy_threshold  = SHORT_BUY_THRESHOLD  × 1.25  = 75
            effective_sell_threshold = SHORT_SELL_THRESHOLD × 1.25  = -75

        Args:
            indicators:      dict from TechnicalIndicators.compute_all()
            ml_prediction:   float in [0, 1] (BUY probability from ShortTrendModel)
            is_extended:     True if current bar is in extended hours
            liquidity_score: Volume_extended / Volume_regular
            gap_type:        'gap_up', 'gap_down', or 'none'
            age_in_minutes:  age of the most-recent data point in minutes
            neutral_probability: calibrated base rate treated as ML-neutral;
                defaults to 0.5 for legacy models and fallbacks

        Returns:
            {
              "score":               float,
              "signal":              str,
              "confidence":          float,
              "breakdown":           dict,
              "weight":              float,
              "ml_score":            float,
              "indicator_score":     float,
              "liquidity_adjusted":  bool,
              "gap_type":            str,
              "macro_override_applied": bool,  # set by caller after override check
            }
        """
        try:
            # ── Indicator contributions ────────────────────────────────────────
            indicator_score, breakdown = self.compute_indicator_score(
                indicators, is_extended, liquidity_score, gap_type, gap_momentum
            )
            liquidity_adjusted = liquidity_score < settings.LIQUIDITY_SCORE_THRESHOLD

            # ── ML contribution ────────────────────────────────────────────────
            # A calibrated rare-event probability such as 0.08 is normal, not
            # evidence of a bearish market. Center on the measured base rate
            # while retaining the existing full range at p=0 and p=1.
            p = min(1.0, max(0.0, float(ml_prediction)))
            base = min(0.99, max(0.01, float(neutral_probability)))
            if p >= base:
                ml_score = (p - base) / (1.0 - base) * (_ML_WEIGHT / 2.0)
            else:
                ml_score = (p - base) / base * (_ML_WEIGHT / 2.0)
            breakdown["ml_score"] = ml_score
            breakdown["ml_neutral_probability"] = base

            raw_score = indicator_score + ml_score

            # ── Time-decay ─────────────────────────────────────────────────────
            # base_weight = 1.0 for both regular and extended sessions.
            # Extended-hours uncertainty is already captured by the 0.5
            # indicator multiplier inside compute_indicator_score; applying
            # a second 0.5 here would make the threshold unreachable for any
            # extended-hours setup regardless of signal strength.
            # Weight(t) = 1.0 × exp(−LAMBDA_SHORT × age_in_minutes)
            base_weight = 1.0
            weight = base_weight * math.exp(
                -settings.LAMBDA_SHORT * max(0.0, age_in_minutes)
            )
            total_score = raw_score * weight

            # ── Clamp ──────────────────────────────────────────────────────────
            total_score = max(_SCORE_MIN, min(_SCORE_MAX, total_score))

            # ── Signal thresholds (with liquidity adjustment) ──────────────────
            # If thin liquidity: thresholds × 1.25
            if liquidity_adjusted:
                buy_thresh = settings.SHORT_BUY_THRESHOLD * 1.25
                sell_thresh = settings.SHORT_SELL_THRESHOLD * 1.25
            else:
                buy_thresh = settings.SHORT_BUY_THRESHOLD
                sell_thresh = settings.SHORT_SELL_THRESHOLD

            if total_score > buy_thresh:
                signal = "buy"
            elif total_score < sell_thresh:
                signal = "sell"
            else:
                signal = "neutral"

            confidence = round(abs(total_score) / 100.0, 4)

            return {
                "score": round(total_score, 4),
                "signal": signal,
                "confidence": confidence,
                "breakdown": breakdown,
                "weight": round(weight, 6),
                "ml_score": round(ml_score, 4),
                "indicator_score": round(indicator_score, 4),
                "liquidity_adjusted": liquidity_adjusted,
                "gap_type": gap_type,
                "gap_momentum": gap_momentum,
                "neutral_probability": base,
                "macro_override_applied": False,  # will be set by caller
            }

        except Exception as exc:
            logger.error("ShortTrendGauge.compute_score error: %s", exc)
            return {
                "score": 0.0,
                "signal": "neutral",
                "confidence": 0.0,
                "breakdown": {},
                "weight": 1.0,
                "ml_score": 0.0,
                "indicator_score": 0.0,
                "liquidity_adjusted": False,
                "gap_type": gap_type,
                "macro_override_applied": False,
            }
