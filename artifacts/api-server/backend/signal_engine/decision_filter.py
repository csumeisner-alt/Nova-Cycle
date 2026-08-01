"""
NovaCycle Decision Filter
=========================
Applies post-gauge decision-layer filters to BUY/SELL signals for VOO-only
trading. This module intentionally does not touch ingestion, ML, indicators,
UI, or database schema — it only refines the BUY/SELL decision using the
existing context produced by those layers.

Implemented upgrades:
  1. Volatility regime filtering
  2. Gap-type filtering
  3. Liquidity-class filtering
  4. Confidence divergence suppression
  5. Cycle-quality scoring
"""

import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


class DecisionFilter:
    """
    Filter and score VOO trading signals after the long/short gauges have
    computed their raw score and signal.
    """

    _REGIME_SCORES = {
        "calm": 0.25,
        "trending": 0.2,
        "compressed": -0.15,
        "macro_shock": -0.25,
    }

    _LIQUIDITY_SCORES = {
        "high": 0.15,
        "normal": 0.0,
        "low": -0.2,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Input helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def infer_volatility_regime(indicators: Dict[str, Any]) -> str:
        """
        Infer the current volatility regime from the indicator dict produced
        by TechnicalIndicators.compute_all(). This reuses the upgraded
        indicators (atr_compression_score, trend_strength_index) and the VIX
        regime without modifying the indicators module.
        """
        try:
            latest = indicators.get("latest", {})
            vix_regime = str(latest.get("vix_regime", "NORMAL")).upper()
            atr_compression = latest.get("atr_compression_score")
            trend_strength = latest.get("trend_strength_index")

            if vix_regime == "EXTREME":
                return "macro_shock"
            if vix_regime == "HIGH":
                return "trending"
            if atr_compression is not None and float(atr_compression) < 0.2:
                return "compressed"
            if trend_strength is not None and float(trend_strength) > 0.7:
                return "trending"
            return "calm"
        except Exception as exc:
            logger.error("DecisionFilter.infer_volatility_regime error: %s", exc)
            return "calm"

    @staticmethod
    def classify_liquidity(liquidity_score: float) -> str:
        """Classify a liquidity score into high / normal / low."""
        try:
            score = float(liquidity_score)
            if score >= 1.0:
                return "high"
            if score >= 0.5:
                return "normal"
            return "low"
        except Exception:
            return "low"

    @staticmethod
    def _nonzero_values(history: List[Dict[str, Any]], key: str) -> List[float]:
        """Extract non-zero/non-None values for a key from confidence history."""
        values = []
        for entry in history:
            try:
                val = entry.get(key)
                if val is None:
                    continue
                val = float(val)
                if val != 0.0:
                    values.append(val)
            except Exception:
                continue
        return values

    def compute_confidence_metrics(
        self, confidence_history: List[Dict[str, Any]]
    ) -> tuple[float, float, float]:
        """
        Compute (confidence_long, confidence_short, confidence_momentum) from
        recent confidence history. Only non-zero values are used so that
        interleaved long/short snapshots do not pollute the momentum estimate.
        """
        try:
            long_values = self._nonzero_values(confidence_history, "long_buy_confidence")
            short_values = self._nonzero_values(confidence_history, "short_buy_confidence")

            confidence_long = long_values[-1] if long_values else 0.5
            confidence_short = short_values[-1] if short_values else 0.5

            confidence_momentum = 0.0
            if len(short_values) >= 2:
                confidence_momentum = short_values[-1] - short_values[-2]

            return confidence_long, confidence_short, confidence_momentum
        except Exception as exc:
            logger.error("DecisionFilter.compute_confidence_metrics error: %s", exc)
            return 0.5, 0.5, 0.0

    def _has_divergence(self, confidence_history: List[Dict[str, Any]]) -> bool:
        """
        True when long-trend confidence is rising but short-trend confidence
        is falling between the two most recent meaningful snapshots.
        """
        try:
            long_values = self._nonzero_values(confidence_history, "long_buy_confidence")
            short_values = self._nonzero_values(confidence_history, "short_buy_confidence")

            if len(long_values) < 2 or len(short_values) < 2:
                return False

            return long_values[-1] > long_values[-2] and short_values[-1] < short_values[-2]
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Cycle quality scoring
    # ──────────────────────────────────────────────────────────────────────────

    def compute_cycle_quality_score(
        self,
        volatility_regime: str,
        gap_type: str,
        gap_percent: float,
        gap_momentum: Optional[float],
        liquidity_class: str,
        confidence_momentum: float,
    ) -> float:
        """
        Compute a [0, 1] score for the expected quality of the upcoming cycle.

        Higher score = more favorable conditions for a BUY signal.
        Lower score = conditions that favor caution or SELL prioritization.
        """
        try:
            score = 0.5
            score += self._REGIME_SCORES.get(volatility_regime, 0.0)
            score += self._LIQUIDITY_SCORES.get(liquidity_class, 0.0)

            gap_pct = float(gap_percent) if gap_percent is not None else 0.0
            macro_threshold = float(settings.MACRO_GAP_THRESHOLD)
            momentum_threshold = float(settings.GAP_MOMENTUM_THRESHOLD)

            if gap_pct < -macro_threshold:
                # Negative macro gap hurts BUY quality
                score -= 0.25
            elif gap_pct > macro_threshold:
                # Positive continuation gap helps if it is following through
                if gap_momentum is not None and float(gap_momentum) >= momentum_threshold:
                    score += 0.2
                else:
                    score += 0.05

            cm = float(confidence_momentum)
            if cm > 0:
                score += min(0.15, cm)
            else:
                score += max(-0.15, cm)

            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.error("DecisionFilter.compute_cycle_quality_score error: %s", exc)
            return 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # Main signal evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        signal_type: str,
        score: float,
        ml_confidence: float,
        indicators: Dict[str, Any],
        latest_candle: Dict[str, Any],
        liquidity_score: float,
        gap_momentum: Optional[float],
        confidence_history: List[Dict[str, Any]],
        data_quality_degraded: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply all decision-layer filters to a raw gauge signal.

        Args:
            signal_type: 'buy', 'sell', or 'neutral' from the gauge.
            score: gauge score in [-100, 100].
            ml_confidence: ML probability in [0, 1].
            indicators: dict from TechnicalIndicators.compute_all().
            latest_candle: dict-like row for the current candle.
            liquidity_score: current liquidity score.
            gap_momentum: optional gap follow-through %.
            confidence_history: recent confidence snapshots (each with
                long_buy_confidence / short_buy_confidence keys).

        Returns:
            dict with allowed, final_signal, priority_boost, decision_penalty,
            conviction_tier_cap, cycle_quality_score, volatility_regime,
            liquidity_class, confidence metrics, filter_flags, and a
            human-readable reason.
        """
        try:
            signal = str(signal_type).lower().strip()
            if signal not in ("buy", "sell"):
                return {
                    "allowed": True,
                    "final_signal": signal,
                    "priority_boost": 0.0,
                    "cycle_quality_score": 0.5,
                    "volatility_regime": "calm",
                    "liquidity_class": "normal",
                    "confidence_long": 0.5,
                    "confidence_short": 0.5,
                    "confidence_momentum": 0.0,
                    "filter_flags": {},
                    "reason": "Signal is neutral; no filters applied.",
                    "decision_penalty": 0.0,
                    "conviction_tier_cap": None,
                }

            volatility_regime = self.infer_volatility_regime(indicators)
            liquidity_class = self.classify_liquidity(liquidity_score)
            (
                confidence_long,
                confidence_short,
                confidence_momentum,
            ) = self.compute_confidence_metrics(confidence_history)
            divergence = self._has_divergence(confidence_history)

            gap_type = str(latest_candle.get("gap_type", "none")).lower()
            gap_percent = float(latest_candle.get("gap_percent", 0.0) or 0.0)

            cycle_quality_score = self.compute_cycle_quality_score(
                volatility_regime=volatility_regime,
                gap_type=gap_type,
                gap_percent=gap_percent,
                gap_momentum=gap_momentum,
                liquidity_class=liquidity_class,
                confidence_momentum=confidence_momentum,
            )

            min_quality = float(getattr(settings, "DECISION_BUY_MIN_CYCLE_QUALITY", 0.6))

            filter_flags = {
                "volatility_regime": volatility_regime,
                "liquidity_class": liquidity_class,
                "gap_type": gap_type,
                "gap_percent": round(gap_percent, 4),
                "divergence": divergence,
                "cycle_quality_score": round(cycle_quality_score, 4),
                "confidence_momentum": round(confidence_momentum, 4),
                "liquidity_score": round(float(liquidity_score), 4),
                "data_quality_degraded": bool(data_quality_degraded),
            }

            # Data-quality degradation is a safety stop.  A malformed or
            # incomplete candle set must never be promoted to an Opportunity
            # merely because the raw gauge happened to cross its threshold.
            if data_quality_degraded:
                return self._blocked(
                    cycle_quality_score,
                    volatility_regime,
                    liquidity_class,
                    confidence_long,
                    confidence_short,
                    confidence_momentum,
                    filter_flags,
                    "Signal blocked: market data quality is degraded.",
                )

            if signal == "buy":
                return self._evaluate_buy(
                    volatility_regime=volatility_regime,
                    liquidity_class=liquidity_class,
                    gap_type=gap_type,
                    gap_percent=gap_percent,
                    gap_momentum=gap_momentum,
                    cycle_quality_score=cycle_quality_score,
                    min_quality=min_quality,
                    divergence=divergence,
                    confidence_momentum=confidence_momentum,
                    confidence_long=confidence_long,
                    confidence_short=confidence_short,
                    filter_flags=filter_flags,
                )

            return self._evaluate_sell(
                volatility_regime=volatility_regime,
                liquidity_class=liquidity_class,
                gap_type=gap_type,
                gap_percent=gap_percent,
                gap_momentum=gap_momentum,
                cycle_quality_score=cycle_quality_score,
                min_quality=min_quality,
                confidence_momentum=confidence_momentum,
                confidence_long=confidence_long,
                confidence_short=confidence_short,
                filter_flags=filter_flags,
            )

        except Exception as exc:
            logger.error("DecisionFilter.evaluate error: %s", exc)
            signal = str(signal_type).lower().strip()
            return {
                "allowed": True,
                "final_signal": signal if signal in ("buy", "sell") else "neutral",
                "priority_boost": 0.0,
                "cycle_quality_score": 0.5,
                "volatility_regime": "calm",
                "liquidity_class": "normal",
                "confidence_long": 0.5,
                "confidence_short": 0.5,
                "confidence_momentum": 0.0,
                "filter_flags": {"error": str(exc)},
                "reason": f"Filter error (defaulting to allowed): {exc}",
                "decision_penalty": 0.0,
                "conviction_tier_cap": None,
            }

    # ──────────────────────────────────────────────────────────────────────────
    # BUY-specific rules
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_buy(
        self,
        volatility_regime: str,
        liquidity_class: str,
        gap_type: str,
        gap_percent: float,
        gap_momentum: Optional[float],
        cycle_quality_score: float,
        min_quality: float,
        divergence: bool,
        confidence_momentum: float,
        confidence_long: float,
        confidence_short: float,
        filter_flags: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Severe macro shocks remain a hard safety block.  Compressed
        # volatility is only a downgrade: a strong setup can still be useful
        # as an Opportunity ahead of a breakout.
        if volatility_regime == "macro_shock":
            return self._blocked(
                cycle_quality_score,
                volatility_regime,
                liquidity_class,
                confidence_long,
                confidence_short,
                confidence_momentum,
                filter_flags,
                f"BUY blocked: volatility_regime={volatility_regime} is unfavorable.",
            )

        penalty = 0.0
        downgrade_reasons: List[str] = []

        if volatility_regime == "compressed":
            penalty += 0.08
            downgrade_reasons.append("compressed volatility")

        # A negative gap is a warning, not proof that a reversal setup is
        # invalid.  Keep the signal but cap it at Opportunity.
        if gap_percent < -float(settings.MACRO_GAP_THRESHOLD):
            penalty += 0.08
            downgrade_reasons.append(f"negative macro gap ({gap_percent:+.2f}%)")

        # Extremely thin liquidity is still a hard block.  Moderate low
        # liquidity is a downgrade so valid regular-session setups are not
        # silently discarded.
        if liquidity_class == "low" and (
            float(filter_flags.get("liquidity_score", 1.0))
            < float(settings.LIQUIDITY_SCORE_THRESHOLD)
        ):
            return self._blocked(
                cycle_quality_score,
                volatility_regime,
                liquidity_class,
                confidence_long,
                confidence_short,
                confidence_momentum,
                filter_flags,
                "BUY blocked: extremely low liquidity.",
            )

        if liquidity_class == "low":
            penalty += 0.10
            downgrade_reasons.append("moderately low liquidity")

        # Divergence and negative momentum reduce conviction, but reversals
        # often begin with exactly this disagreement.
        if divergence:
            penalty += 0.08
            downgrade_reasons.append(
                "long confidence rising while short confidence is falling"
            )

        if confidence_momentum < 0:
            penalty += min(0.08, abs(float(confidence_momentum)))
            downgrade_reasons.append("confidence momentum is negative")

        # Cycle quality is a ranking/conviction input.  Only severe safety
        # conditions above block; marginal quality keeps the signal as an
        # Opportunity and applies a notification penalty.
        if cycle_quality_score < min_quality:
            penalty += min(0.10, max(0.0, min_quality - cycle_quality_score))
            downgrade_reasons.append(
                f"cycle quality {cycle_quality_score:.3f} is below {min_quality:.3f}"
            )

        # Small priority boost for positive continuation gaps with follow-through
        priority_boost = 0.0
        if gap_percent > float(settings.MACRO_GAP_THRESHOLD) and gap_momentum is not None:
            if float(gap_momentum) >= float(settings.GAP_MOMENTUM_THRESHOLD):
                priority_boost += 0.05

        if downgrade_reasons:
            reason = (
                "BUY allowed as Opportunity; conviction reduced by "
                + ", ".join(downgrade_reasons)
                + "."
            )
        else:
            reason = "BUY allowed: all decision filters passed."

        return self._allowed(
            "buy",
            priority_boost,
            cycle_quality_score,
            volatility_regime,
            liquidity_class,
            confidence_long,
            confidence_short,
            confidence_momentum,
            filter_flags,
            reason,
            decision_penalty=penalty,
            conviction_tier_cap="opportunity" if downgrade_reasons else None,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SELL-specific rules
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_sell(
        self,
        volatility_regime: str,
        liquidity_class: str,
        gap_type: str,
        gap_percent: float,
        gap_momentum: Optional[float],
        cycle_quality_score: float,
        min_quality: float,
        confidence_momentum: float,
        confidence_long: float,
        confidence_short: float,
        filter_flags: Dict[str, Any],
    ) -> Dict[str, Any]:
        priority_boost = 0.0
        reasons: List[str] = []
        penalty = 0.0

        # Increase SELL priority in unfavorable conditions
        if volatility_regime == "macro_shock":
            priority_boost += 0.1
            reasons.append("macro_shock regime")

        if liquidity_class == "low":
            priority_boost += 0.1
            if float(filter_flags.get("liquidity_score", 1.0)) < float(
                settings.LIQUIDITY_SCORE_THRESHOLD
            ):
                return self._blocked(
                    cycle_quality_score,
                    volatility_regime,
                    liquidity_class,
                    confidence_long,
                    confidence_short,
                    confidence_momentum,
                    filter_flags,
                    "SELL blocked: extremely low liquidity.",
                )
            penalty += 0.10
            reasons.append("low liquidity")

        if cycle_quality_score < min_quality:
            priority_boost += 0.1
            penalty += min(0.10, max(0.0, min_quality - cycle_quality_score))
            reasons.append("low cycle quality")

        # Block SELL during strong positive continuation gaps unless momentum flips
        if gap_percent > float(settings.MACRO_GAP_THRESHOLD):
            if confidence_momentum < 0:
                reasons.append("strong positive gap but confidence momentum flipped")
            else:
                return self._blocked(
                    cycle_quality_score,
                    volatility_regime,
                    liquidity_class,
                    confidence_long,
                    confidence_short,
                    confidence_momentum,
                    filter_flags,
                    "SELL blocked: strong positive continuation gap without momentum flip.",
                )

        reason = "SELL allowed."
        if reasons:
            reason += " Priority increased: " + ", ".join(reasons) + "."

        if penalty:
            reason += " Kept as Opportunity with a conviction penalty."

        return self._allowed(
            "sell",
            min(0.3, priority_boost),
            cycle_quality_score,
            volatility_regime,
            liquidity_class,
            confidence_long,
            confidence_short,
            confidence_momentum,
            filter_flags,
            reason,
            decision_penalty=penalty,
            conviction_tier_cap="opportunity" if penalty else None,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Result builders
    # ──────────────────────────────────────────────────────────────────────────

    def _blocked(
        self,
        cycle_quality_score: float,
        volatility_regime: str,
        liquidity_class: str,
        confidence_long: float,
        confidence_short: float,
        confidence_momentum: float,
        filter_flags: Dict[str, Any],
        reason: str,
        decision_penalty: float = 0.0,
        conviction_tier_cap: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "allowed": False,
            "final_signal": "neutral",
            "priority_boost": 0.0,
            "cycle_quality_score": cycle_quality_score,
            "volatility_regime": volatility_regime,
            "liquidity_class": liquidity_class,
            "confidence_long": confidence_long,
            "confidence_short": confidence_short,
            "confidence_momentum": confidence_momentum,
            "filter_flags": filter_flags,
            "reason": reason,
            "decision_penalty": decision_penalty,
            "conviction_tier_cap": conviction_tier_cap,
        }

    def _allowed(
        self,
        final_signal: str,
        priority_boost: float,
        cycle_quality_score: float,
        volatility_regime: str,
        liquidity_class: str,
        confidence_long: float,
        confidence_short: float,
        confidence_momentum: float,
        filter_flags: Dict[str, Any],
        reason: str,
        decision_penalty: float = 0.0,
        conviction_tier_cap: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "allowed": True,
            "final_signal": final_signal,
            "priority_boost": priority_boost,
            "cycle_quality_score": cycle_quality_score,
            "volatility_regime": volatility_regime,
            "liquidity_class": liquidity_class,
            "confidence_long": confidence_long,
            "confidence_short": confidence_short,
            "confidence_momentum": confidence_momentum,
            "filter_flags": filter_flags,
            "reason": reason,
            "decision_penalty": decision_penalty,
            "conviction_tier_cap": conviction_tier_cap,
        }
