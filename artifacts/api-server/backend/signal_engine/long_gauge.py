"""
NovaCycle Long-Trend Gauge
===========================
Aggregates long-term technical indicators and ML model output into
a single score in the range [-100, +100].

Score composition:
  indicator_score  (max ±30 total from individual ±10 contributions)
  ml_score         = ml_prediction × 80 − 40   → maps [0,1] to [-40,+40]
  total_score      = indicator_score + ml_score  (clamped to [-100, +100])

BUY  threshold : total_score > +70
SELL threshold : total_score < -70
NEUTRAL        : otherwise

Time-decay:
  Weight(t) = exp(−LAMBDA_LONG × age_in_days)
  The score is multiplied by this weight before thresholding; it reflects
  how stale the input data is (fresh data → weight ≈ 1.0).

Rules:
  - Extended-hours candles MUST NOT be used here.
  - SMA50/SMA200, MACD, ADX, VIX regime use regular-hours candles only.
"""

import logging
import math

from config import settings

logger = logging.getLogger(__name__)

# Individual indicator contribution caps (±10 each, total cap ±30 enforced)
_IND_CAP = 10.0
# ML score range: ml_prediction in [0,1] → ml_score in [-40, +40]
_ML_WEIGHT = 80.0
_ML_OFFSET = 40.0
# Total score clamp
_SCORE_MIN = -100.0
_SCORE_MAX = 100.0


class LongTrendGauge:
    """Compute long-trend gauge score from regular-hours indicators + ML."""

    # ──────────────────────────────────────────────────────────────────────────
    # Indicator scoring
    # ──────────────────────────────────────────────────────────────────────────

    def compute_indicator_score(self, indicators: dict) -> tuple[float, dict]:
        """
        Compute the indicator contribution to the long-trend score.

        Individual contributions (each capped at ±10):
          SMA cross:
            SMA50 > SMA200  → +10  (golden cross: bullish)
            SMA50 < SMA200  → -10  (death cross: bearish)
            SMA unavailable →   0

          MACD histogram:
            histogram > 0   → +10  (momentum building)
            histogram < 0   → -10  (momentum fading)
            unavailable     →   0

          ADX (trend strength amplifier):
            ADX > 25        → add ±5 in the direction of the existing score
                              (amplifies signal when market is trending)
            ADX < 25        → no amplification

          VIX regime penalty:
            LOW    →  0
            NORMAL →  0
            HIGH   → -5
            EXTREME→ -10

        Returns:
            (total_indicator_score: float, breakdown: dict)
        """
        breakdown: dict = {}
        total = 0.0

        latest = indicators.get("latest", {})

        # ── SMA50 / SMA200 cross ───────────────────────────────────────────────
        sma50 = latest.get("sma50")
        sma200 = latest.get("sma200")
        if sma50 is not None and sma200 is not None and sma200 != 0:
            sma_score = _IND_CAP if sma50 > sma200 else -_IND_CAP
        else:
            sma_score = 0.0
        breakdown["sma_cross"] = sma_score
        total += sma_score

        # ── MACD histogram ─────────────────────────────────────────────────────
        macd_hist = latest.get("macd_histogram")
        if macd_hist is not None:
            macd_score = _IND_CAP if macd_hist > 0 else -_IND_CAP
        else:
            macd_score = 0.0
        breakdown["macd_histogram"] = macd_score
        total += macd_score

        # ── ADX amplifier ──────────────────────────────────────────────────────
        adx = latest.get("adx")
        adx_score = 0.0
        if adx is not None and adx > 25.0:
            # Amplify in the direction of the current score
            adx_score = 5.0 if total >= 0 else -5.0
        breakdown["adx_amplifier"] = adx_score
        total += adx_score

        # ── VIX regime penalty ─────────────────────────────────────────────────
        vix_regime = latest.get("vix_regime", "NORMAL")
        vix_penalty = {"LOW": 0.0, "NORMAL": 0.0, "HIGH": -5.0, "EXTREME": -10.0}.get(
            str(vix_regime).upper(), 0.0
        )
        breakdown["vix_penalty"] = vix_penalty
        total += vix_penalty

        return total, breakdown

    # ──────────────────────────────────────────────────────────────────────────
    # Full score computation
    # ──────────────────────────────────────────────────────────────────────────

    def compute_score(
        self,
        indicators: dict,
        ml_prediction: float,
        age_in_days: float = 0.0,
    ) -> dict:
        """
        Compute the final long-trend gauge score.

        Formula:
          indicator_score = Σ indicator contributions   (see compute_indicator_score)
          ml_score        = ml_prediction × 80 − 40
                            maps [0 → -40, 0.5 → 0, 1.0 → +40]
          raw_score       = indicator_score + ml_score

          # Time-decay weighting
          Weight(t)       = exp(−LAMBDA_LONG × age_in_days)
          total_score     = raw_score × Weight(t)
          total_score     = clamp(total_score, −100, +100)

        Signal rules:
          score >  LONG_BUY_THRESHOLD  (+70) → 'buy'
          score <  LONG_SELL_THRESHOLD (−70) → 'sell'
          otherwise                          → 'neutral'

        Confidence:
          confidence = |total_score| / 100.0

        Args:
            indicators:    dict from TechnicalIndicators.compute_all()
            ml_prediction: float in [0, 1] (BUY probability from LongTrendModel)
            age_in_days:   age of the most-recent data point in days

        Returns:
            {
              "score":        float,
              "signal":       str,    # 'buy' | 'sell' | 'neutral'
              "confidence":   float,  # [0, 1]
              "breakdown":    dict,
              "weight":       float,
              "ml_score":     float,
              "indicator_score": float,
            }
        """
        try:
            # ── Indicator contributions ────────────────────────────────────────
            indicator_score, breakdown = self.compute_indicator_score(indicators)

            # ── ML contribution ────────────────────────────────────────────────
            # ml_score = ml_prediction × 80 − 40  → [-40, +40]
            ml_score = float(ml_prediction) * _ML_WEIGHT - _ML_OFFSET
            breakdown["ml_score"] = ml_score

            raw_score = indicator_score + ml_score

            # ── Time-decay weighting ───────────────────────────────────────────
            # Weight(t) = exp(−LAMBDA_LONG × age_in_days)
            weight = math.exp(-settings.LAMBDA_LONG * max(0.0, age_in_days))
            total_score = raw_score * weight

            # ── Clamp ──────────────────────────────────────────────────────────
            total_score = max(_SCORE_MIN, min(_SCORE_MAX, total_score))

            # ── Signal classification ──────────────────────────────────────────
            if total_score > settings.LONG_BUY_THRESHOLD:
                signal = "buy"
            elif total_score < settings.LONG_SELL_THRESHOLD:
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
            }

        except Exception as exc:
            logger.error("LongTrendGauge.compute_score error: %s", exc)
            return {
                "score": 0.0,
                "signal": "neutral",
                "confidence": 0.0,
                "breakdown": {},
                "weight": 1.0,
                "ml_score": 0.0,
                "indicator_score": 0.0,
            }
