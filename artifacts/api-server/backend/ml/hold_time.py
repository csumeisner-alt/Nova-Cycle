"""
NovaCycle Hold-Time Prediction Engine
======================================
Estimates expected hold duration for an active BUY signal using
a rule-based system layered on top of the gauge scores and indicators.

No ML model – pure heuristic with configurable multipliers.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HoldTimePredictionEngine:
    """
    Estimate how long to hold a VOO position opened on a BUY signal.

    Base hold-time rules:
      long_score  > 70  → base = 15 days  (21_600 minutes)
      short_score > 50  → base = 2 hours  (    120 minutes)
      both triggered    → take the LONGER of the two (long dominates)
      neither triggered → default = 4 hours (240 minutes)

    Adjustments (multiplicative, applied in order):
      VIX regime:
        HIGH    × 0.7
        EXTREME × 0.5
        LOW     × 1.1
        NORMAL  × 1.0

      ADX (trend strength):
        ADX > 25 (trending) × 1.2
        ADX < 15 (choppy)   × 0.6

      Bollinger bandwidth:
        narrow  (< 0.02)    × 0.8
        wide    (> 0.05)    × 1.3
    """

    # Base times in minutes
    _BASE_LONG_MINUTES: float = 15 * 24 * 60   # 15 trading days ≈ 21,600 min
    _BASE_SHORT_MINUTES: float = 120.0           # 2 hours
    _BASE_DEFAULT_MINUTES: float = 240.0         # 4 hours fallback

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def estimate_hold_time(
        self,
        indicators: dict,
        long_score: float,
        short_score: float,
        vix_regime: str,
    ) -> dict:
        """
        Estimate expected hold time for the current BUY signal context.

        Args:
            indicators:  Output of TechnicalIndicators.compute_all() (uses 'latest' sub-dict)
            long_score:  LongTrendGauge score  (-100 to +100)
            short_score: ShortTrendGauge score (-100 to +100)
            vix_regime:  One of 'LOW', 'NORMAL', 'HIGH', 'EXTREME'

        Returns:
            {
              "minutes":        float,
              "human_readable": str,
              "confidence":     float,   # 0.0 – 1.0
              "reasoning":      list[str]
            }
        """
        reasoning: list[str] = []
        confidence_factors: list[float] = []

        # ── Step 1: Determine base hold time ──────────────────────────────────
        long_triggered = long_score > 70.0
        short_triggered = short_score > 50.0  # mirrors SHORT_BUY_THRESHOLD

        if long_triggered and short_triggered:
            base_minutes = self._BASE_LONG_MINUTES
            reasoning.append(
                f"Both long ({long_score:.1f}) and short ({short_score:.1f}) "
                "signals active → using long base: 15 days"
            )
            confidence_factors.append(0.85)
        elif long_triggered:
            base_minutes = self._BASE_LONG_MINUTES
            reasoning.append(
                f"Long-trend BUY active (score={long_score:.1f}) → base: 15 days"
            )
            confidence_factors.append(0.75)
        elif short_triggered:
            base_minutes = self._BASE_SHORT_MINUTES
            reasoning.append(
                f"Short-trend BUY active (score={short_score:.1f}) → base: 2 hours"
            )
            confidence_factors.append(0.70)
        else:
            base_minutes = self._BASE_DEFAULT_MINUTES
            reasoning.append(
                "No active BUY signal above threshold → default: 4 hours"
            )
            confidence_factors.append(0.40)

        minutes = base_minutes

        # ── Step 2: VIX regime adjustment ─────────────────────────────────────
        vix_multiplier = self._vix_multiplier(vix_regime)
        if vix_multiplier != 1.0:
            reasoning.append(
                f"VIX regime '{vix_regime}' → multiplier × {vix_multiplier:.1f}"
            )
        minutes *= vix_multiplier
        confidence_factors.append(self._vix_confidence(vix_regime))

        # ── Step 3: ADX adjustment ────────────────────────────────────────────
        latest = indicators.get("latest", {})
        adx_val: Optional[float] = latest.get("adx")

        if adx_val is not None:
            adx_mult, adx_note = self._adx_multiplier(adx_val)
            if adx_mult != 1.0:
                reasoning.append(f"ADX={adx_val:.1f} ({adx_note}) → × {adx_mult:.1f}")
            minutes *= adx_mult
            confidence_factors.append(0.80 if adx_mult != 1.0 else 0.70)
        else:
            reasoning.append("ADX unavailable → no ADX adjustment")

        # ── Step 4: Bollinger bandwidth adjustment ────────────────────────────
        bb_bw: Optional[float] = latest.get("bb_bandwidth")

        if bb_bw is not None:
            bw_mult, bw_note = self._bandwidth_multiplier(bb_bw)
            if bw_mult != 1.0:
                reasoning.append(
                    f"Bollinger bandwidth={bb_bw:.4f} ({bw_note}) → × {bw_mult:.1f}"
                )
            minutes *= bw_mult
            confidence_factors.append(0.75 if bw_mult != 1.0 else 0.70)
        else:
            reasoning.append("Bollinger bandwidth unavailable → no bandwidth adjustment")

        # ── Step 5: Floor / ceiling ───────────────────────────────────────────
        minutes = max(15.0, min(minutes, self._BASE_LONG_MINUTES * 2))

        # ── Step 6: Aggregate confidence ──────────────────────────────────────
        confidence = float(
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.5
        )
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        return {
            "minutes": round(minutes, 1),
            "human_readable": self._format_duration(minutes),
            "confidence": confidence,
            "reasoning": reasoning,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _vix_multiplier(regime: str) -> float:
        return {
            "LOW":     1.1,
            "NORMAL":  1.0,
            "HIGH":    0.7,
            "EXTREME": 0.5,
        }.get(regime.upper(), 1.0)

    @staticmethod
    def _vix_confidence(regime: str) -> float:
        return {
            "LOW":     0.85,
            "NORMAL":  0.75,
            "HIGH":    0.65,
            "EXTREME": 0.50,
        }.get(regime.upper(), 0.70)

    @staticmethod
    def _adx_multiplier(adx: float) -> tuple[float, str]:
        if adx > 25.0:
            return 1.2, "trending"
        elif adx < 15.0:
            return 0.6, "choppy"
        return 1.0, "neutral"

    @staticmethod
    def _bandwidth_multiplier(bw: float) -> tuple[float, str]:
        if bw < 0.02:
            return 0.8, "narrow"
        elif bw > 0.05:
            return 1.3, "wide"
        return 1.0, "normal"

    @staticmethod
    def _format_duration(minutes: float) -> str:
        """Convert raw minutes into a human-readable string."""
        try:
            total_min = int(round(minutes))
            if total_min >= 1440:                      # ≥ 1 day
                days = total_min // 1440
                hours = (total_min % 1440) // 60
                if hours > 0:
                    return f"{days}d {hours}h"
                return f"{days} day{'s' if days != 1 else ''}"
            elif total_min >= 60:                      # ≥ 1 hour
                hours = total_min // 60
                mins = total_min % 60
                if mins > 0:
                    return f"{hours}h {mins}m"
                return f"{hours} hour{'s' if hours != 1 else ''}"
            else:
                return f"{total_min} minute{'s' if total_min != 1 else ''}"
        except Exception:
            return f"{minutes:.0f} minutes"
