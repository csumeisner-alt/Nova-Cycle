"""
NovaCycle Conviction Tier Evaluator
====================================
Labels every actionable BUY/SELL signal with a conviction tier:

  * ``opportunity``     — the signal passed today's thresholds (baseline tier;
                          every actionable signal is at least this).
  * ``high_conviction`` — the signal ALSO passed regime-aware confirmation:
      1. Favorable volatility regime (calm or trending).
      2. Cycle quality score at/above the high-conviction threshold.
      3. Long- and short-trend gauges agree with the signal direction.
      4. ML confidence at/above the high-conviction threshold and no ML
         fallback (a neutral 0.5 fallback can never earn high conviction).

Robustness rules:
  * Regime-transition downgrade — when the volatility regime changed within
    the recent transition window, conditions are treated as unsettled and the
    tier is capped at ``opportunity``.
  * Tier hysteresis — once a gauge's signal is tiered, it keeps that tier on
    subsequent evaluations unless the inputs move decisively past a wider
    exit threshold, so tiers don't flicker near the boundary.

This layer NEVER suppresses a signal — it only labels. The existing decision
filter / macro override / reliability gate remain the only suppression paths.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TIER_OPPORTUNITY = "opportunity"
TIER_HIGH_CONVICTION = "high_conviction"

# ── Entry thresholds (to EARN high conviction) ──────────────────────────────
FAVORABLE_REGIMES = ("calm", "trending")
MIN_CYCLE_QUALITY = 0.65
MIN_ML_CONFIDENCE = 0.65
# A gauge "agrees" with a buy when its score is above +AGREEMENT_BAND, with a
# sell when below -AGREEMENT_BAND. The band keeps a barely-positive score
# from counting as meaningful agreement.
AGREEMENT_BAND = 10.0

# ── Exit thresholds (to LOSE high conviction — wider, for hysteresis) ───────
EXIT_MIN_CYCLE_QUALITY = 0.55
EXIT_MIN_ML_CONFIDENCE = 0.55
EXIT_AGREEMENT_BAND = 0.0  # agreement is lost only when the gauge flips sign

# ── Regime transition window (seconds). A regime change within this window
#    caps the tier at opportunity. ────────────────────────────────────────────
REGIME_TRANSITION_WINDOW_SECONDS = 3 * 3600


class ConvictionEvaluator:
    """Stateful conviction-tier evaluator (single-process Reserved VM)."""

    def __init__(self) -> None:
        # Regime transition tracking
        self._last_regime: Optional[str] = None
        self._last_regime_change_at: Optional[float] = None
        # Hysteresis: last tier per (gauge_type, signal_type)
        self._last_tier: Dict[tuple, str] = {}

    def reset(self) -> None:
        """Clear all internal state (used by tests)."""
        self._last_regime = None
        self._last_regime_change_at = None
        self._last_tier = {}

    # ──────────────────────────────────────────────────────────────────────
    # Regime transition tracking
    # ──────────────────────────────────────────────────────────────────────

    def observe_regime(self, regime: str, now: Optional[float] = None) -> None:
        """Record the current volatility regime; timestamps changes."""
        now = time.time() if now is None else now
        regime = str(regime or "calm").lower()
        if self._last_regime is None:
            self._last_regime = regime
            # First observation: no transition recorded.
            return
        if regime != self._last_regime:
            self._last_regime = regime
            self._last_regime_change_at = now

    def _regime_recently_changed(self, now: float) -> bool:
        return (
            self._last_regime_change_at is not None
            and (now - self._last_regime_change_at) < REGIME_TRANSITION_WINDOW_SECONDS
        )

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        signal_type: str,
        gauge_type: str,
        volatility_regime: str,
        cycle_quality_score: float,
        ml_confidence: float,
        ml_fallback: bool,
        long_score: float,
        short_score: float,
        now: Optional[float] = None,
        tier_cap: Optional[str] = None,
        tier_cap_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute the conviction tier for an actionable signal.

        Returns:
            {
              "tier":    "opportunity" | "high_conviction" | None (neutral),
              "reasons": [str, ...]   — plain-language explanation,
            }
        Never raises; on error returns opportunity with an error reason.
        """
        try:
            now = time.time() if now is None else now
            signal = str(signal_type or "").lower().strip()
            regime = str(volatility_regime or "calm").lower()
            self.observe_regime(regime, now=now)

            if signal not in ("buy", "sell"):
                return {"tier": None, "reasons": []}

            key = (str(gauge_type).lower(), signal)
            previous = self._last_tier.get(key)

            entry_ok, entry_reasons = self._check(
                signal, regime, cycle_quality_score, ml_confidence,
                ml_fallback, long_score, short_score,
                MIN_CYCLE_QUALITY, MIN_ML_CONFIDENCE, AGREEMENT_BAND, FAVORABLE_REGIMES,
            )

            # ── Regime-transition downgrade (always caps at opportunity) ──
            if self._regime_recently_changed(now):
                tier = TIER_OPPORTUNITY
                reasons = [
                    "Market regime shifted recently — conditions are unsettled, "
                    "so this signal is capped at Opportunity tier for now."
                ]
                self._last_tier[key] = tier
                return {"tier": tier, "reasons": reasons}

            # Secondary decision-layer concerns can lower conviction without
            # suppressing an otherwise actionable signal.  Apply this cap
            # before hysteresis so a previously high-conviction signal cannot
            # remain high while a current safety/quality penalty is active.
            if tier_cap == TIER_OPPORTUNITY:
                self._last_tier[key] = TIER_OPPORTUNITY
                reasons = [
                    "Secondary market conditions reduced this signal to "
                    "Opportunity tier; it remains actionable but is not "
                    "high conviction."
                ]
                if tier_cap_reason:
                    reasons.append(tier_cap_reason)
                if entry_reasons:
                    reasons.extend(entry_reasons)
                return {"tier": TIER_OPPORTUNITY, "reasons": reasons}

            if previous == TIER_HIGH_CONVICTION:
                # Hysteresis: stay high-conviction unless inputs decisively
                # fall past the wider exit thresholds.
                exit_ok, exit_reasons = self._check(
                    signal, regime, cycle_quality_score, ml_confidence,
                    ml_fallback, long_score, short_score,
                    EXIT_MIN_CYCLE_QUALITY, EXIT_MIN_ML_CONFIDENCE,
                    EXIT_AGREEMENT_BAND, FAVORABLE_REGIMES,
                )
                if exit_ok:
                    tier = TIER_HIGH_CONVICTION
                    reasons = (
                        entry_reasons if entry_ok else [
                            "Holding High-Conviction tier: conditions softened "
                            "slightly but remain above exit thresholds."
                        ]
                    )
                else:
                    tier = TIER_OPPORTUNITY
                    reasons = exit_reasons
            else:
                tier = TIER_HIGH_CONVICTION if entry_ok else TIER_OPPORTUNITY
                reasons = entry_reasons

            self._last_tier[key] = tier
            return {"tier": tier, "reasons": reasons}

        except Exception as exc:
            logger.error("ConvictionEvaluator.evaluate error: %s", exc)
            return {
                "tier": TIER_OPPORTUNITY,
                "reasons": [f"Conviction check error (defaulting to Opportunity): {exc}"],
            }

    # ──────────────────────────────────────────────────────────────────────
    # Criteria check
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check(
        signal: str,
        regime: str,
        cycle_quality_score: float,
        ml_confidence: float,
        ml_fallback: bool,
        long_score: float,
        short_score: float,
        min_quality: float,
        min_ml: float,
        agreement_band: float,
        favorable_regimes: tuple,
    ) -> tuple:
        """Return (all_passed, reasons). Reasons explain passes AND misses."""
        passed: List[str] = []
        missed: List[str] = []

        regime_label = regime.replace("_", " ")
        if regime in favorable_regimes:
            passed.append(f"Market regime is favorable ({regime_label}).")
        else:
            missed.append(f"Market regime is unfavorable ({regime_label}).")

        try:
            quality = float(cycle_quality_score)
        except (TypeError, ValueError):
            quality = 0.0
        if quality >= min_quality:
            passed.append(f"Cycle quality is strong ({quality:.0%}).")
        else:
            missed.append(f"Cycle quality is below the bar ({quality:.0%}).")

        try:
            ls, ss = float(long_score), float(short_score)
        except (TypeError, ValueError):
            ls, ss = 0.0, 0.0
        if signal == "buy":
            agree = ls > agreement_band and ss > agreement_band
        else:
            agree = ls < -agreement_band and ss < -agreement_band
        if agree:
            passed.append("Long- and short-trend models agree with this signal.")
        else:
            missed.append("Long- and short-trend models do not clearly agree.")

        try:
            ml = float(ml_confidence)
        except (TypeError, ValueError):
            ml = 0.0
        # For sells the model's "buy probability" being LOW is confident;
        # use distance from neutral in the signal direction.
        directional_ml = ml if signal == "buy" else (1.0 - ml)
        if not ml_fallback and directional_ml >= min_ml:
            passed.append(f"ML model confidence is strong ({directional_ml:.0%}).")
        elif ml_fallback:
            missed.append("ML model was unavailable (neutral fallback used).")
        else:
            missed.append(f"ML model confidence is modest ({directional_ml:.0%}).")

        if missed:
            return False, [
                "Opportunity tier: " + " ".join(missed)
            ] + (["Met: " + " ".join(passed)] if passed else [])
        return True, ["High conviction: " + " ".join(passed)]
