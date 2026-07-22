"""
NovaCycle Macro Override Safety Layer
=======================================
Prevents conflicting short-term signals from firing against a
strong long-term trend, unless the short-term ML is very confident.

Rules:
  1. If long_score < -70  AND  short_signal == 'buy'   AND  ml_conf ≤ 0.80
       → suppress signal (return 'neutral')

  2. If long_score > +70  AND  short_signal == 'sell'  AND  ml_conf ≤ 0.80
       → suppress signal (return 'neutral')

  High ML confidence (> 0.80) overrides the macro suppression.

This layer is applied BEFORE every short-term signal output.
"""

import logging

logger = logging.getLogger(__name__)

ML_OVERRIDE_THRESHOLD = 0.80   # ML must exceed this to bypass macro suppression
LONG_STRONG_BEAR = -70.0
LONG_STRONG_BULL = +70.0


class MacroOverrideSafety:
    """Apply macro trend safety checks to short-term signals."""

    def apply_override(
        self,
        long_score: float,
        short_signal: str,
        short_ml_confidence: float,
    ) -> dict:
        """
        Evaluate whether a short-term signal should be suppressed.

        Args:
            long_score:           LongTrendGauge score (-100 to +100)
            short_signal:         'buy', 'sell', or 'neutral'
            short_ml_confidence:  Short-trend ML probability in [0, 1]

        Returns:
            {
              "allowed":          bool,   # True = signal passes, False = suppressed
              "override_applied": bool,   # True if suppression was triggered
              "reason":           str     # Human-readable explanation
            }
        """
        try:
            signal = short_signal.lower().strip()
            ml_conf = float(short_ml_confidence)
            ls = float(long_score)

            # ── Rule 1: Strong bearish trend suppresses short BUY ─────────────
            if ls < LONG_STRONG_BEAR and signal == "buy":
                if ml_conf <= ML_OVERRIDE_THRESHOLD:
                    return {
                        "allowed": False,
                        "override_applied": True,
                        "reason": (
                            f"Long-trend strongly bearish (score={ls:.1f} < {LONG_STRONG_BEAR}). "
                            f"Short BUY suppressed (ML confidence {ml_conf:.2%} ≤ {ML_OVERRIDE_THRESHOLD:.0%} threshold)."
                        ),
                    }
                else:
                    return {
                        "allowed": True,
                        "override_applied": False,
                        "reason": (
                            f"Long-trend bearish (score={ls:.1f}) but short ML confidence "
                            f"{ml_conf:.2%} > {ML_OVERRIDE_THRESHOLD:.0%} — override bypassed."
                        ),
                    }

            # ── Rule 2: Strong bullish trend suppresses short SELL ────────────
            if ls > LONG_STRONG_BULL and signal == "sell":
                if ml_conf <= ML_OVERRIDE_THRESHOLD:
                    return {
                        "allowed": False,
                        "override_applied": True,
                        "reason": (
                            f"Long-trend strongly bullish (score={ls:.1f} > {LONG_STRONG_BULL}). "
                            f"Short SELL suppressed (ML confidence {ml_conf:.2%} ≤ {ML_OVERRIDE_THRESHOLD:.0%} threshold)."
                        ),
                    }
                else:
                    return {
                        "allowed": True,
                        "override_applied": False,
                        "reason": (
                            f"Long-trend bullish (score={ls:.1f}) but short ML confidence "
                            f"{ml_conf:.2%} > {ML_OVERRIDE_THRESHOLD:.0%} — override bypassed."
                        ),
                    }

            # ── No suppression ─────────────────────────────────────────────────
            return {
                "allowed": True,
                "override_applied": False,
                "reason": "No macro override condition triggered.",
            }

        except Exception as exc:
            logger.error("MacroOverrideSafety.apply_override error: %s", exc)
            return {
                "allowed": True,
                "override_applied": False,
                "reason": f"Override check error (defaulting to allowed): {exc}",
            }
