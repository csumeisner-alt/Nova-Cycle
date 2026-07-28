"""
NovaCycle Gauge Normalization
=============================
Converts raw gauge output (score in [-100, +100]) into a clean, display-ready
contract for clients:

  confidence_percent : int 0–100 — sigmoid-normalized magnitude, clamped.
  trend              : "UP" | "DOWN" | "NEUTRAL" — sign of the raw score,
                       with a small neutral band around zero.
  display_signal     : "BUY BIAS" | "SELL BIAS" | "NEUTRAL / HOLD" —
                       BUY BIAS when trend UP and confidence ≥ 65,
                       SELL BIAS when trend DOWN and confidence ≥ 65,
                       otherwise NEUTRAL / HOLD.

Any invalid input (None, NaN, inf, non-numeric) returns the neutral defaults
(0, "NEUTRAL", "NEUTRAL / HOLD") — clients never see garbage values.

Product decision — override consistency (Task: mixed messages)
--------------------------------------------------------------
`display_signal` is derived from the raw gauge score, while the legacy
`signal` field passes through the macro override and decision filter. That
meant the app could show "BUY BIAS" while the filtered signal was "neutral"
(e.g. a macro override suppressed the buy). The decided UX is:

  * The actionable label (`display_signal`) must NEVER contradict a signal
    that a safety layer (macro override) forced to neutral. When the macro
    override suppresses the signal, `display_signal` is downgraded to
    "NEUTRAL / HOLD" via `reconcile_display_signal()`.
  * `trend` and `confidence_percent` are NOT changed — they are factual
    readings of the raw gauge (direction and strength) and the gauge UI may
    still show them; only the actionable bias label is neutralized.

Use `reconcile_display_signal()` after computing the filtered signal.
"""

import math

# Sigmoid steepness: raw score is divided by this before the sigmoid so the
# ±100 range maps onto a usable 0–100% spread (raw ±100 → ~2% / ~98%).
SIGMOID_SCALE = 25.0
# Neutral band: |score| ≤ this → trend NEUTRAL.
NEUTRAL_BAND = 5.0
# Display-signal threshold (percent).
SIGNAL_CONFIDENCE_THRESHOLD = 65

TREND_UP = "UP"
TREND_DOWN = "DOWN"
TREND_NEUTRAL = "NEUTRAL"
SIGNAL_BUY = "BUY BIAS"
SIGNAL_SELL = "SELL BIAS"
SIGNAL_HOLD = "NEUTRAL / HOLD"

NEUTRAL_DEFAULTS = {
    "confidence_percent": 0,
    "trend": TREND_NEUTRAL,
    "display_signal": SIGNAL_HOLD,
}


def normalize_gauge_output(raw_score) -> dict:
    """
    Normalize a raw gauge score into the display contract.

    Returns a dict with keys: confidence_percent (int 0–100), trend,
    display_signal. Never raises; invalid input → neutral defaults.
    """
    try:
        score = float(raw_score)
        if math.isnan(score) or math.isinf(score):
            return dict(NEUTRAL_DEFAULTS)
    except (TypeError, ValueError):
        return dict(NEUTRAL_DEFAULTS)

    # Sigmoid on the magnitude → 0.5..1.0, rescaled to 0.0..1.0, clamped.
    sig = 1.0 / (1.0 + math.exp(-abs(score) / SIGMOID_SCALE))
    normalized = (sig - 0.5) * 2.0
    normalized = max(0.0, min(1.0, normalized))
    confidence_percent = int(round(normalized * 100))
    confidence_percent = max(0, min(100, confidence_percent))

    if score > NEUTRAL_BAND:
        trend = TREND_UP
    elif score < -NEUTRAL_BAND:
        trend = TREND_DOWN
    else:
        trend = TREND_NEUTRAL

    if trend == TREND_UP and confidence_percent >= SIGNAL_CONFIDENCE_THRESHOLD:
        display_signal = SIGNAL_BUY
    elif trend == TREND_DOWN and confidence_percent >= SIGNAL_CONFIDENCE_THRESHOLD:
        display_signal = SIGNAL_SELL
    else:
        display_signal = SIGNAL_HOLD

    return {
        "confidence_percent": confidence_percent,
        "trend": trend,
        "display_signal": display_signal,
    }


def reconcile_display_signal(normalized: dict, final_signal,
                             macro_override_applied) -> dict:
    """
    Ensure the display contract never contradicts an override-suppressed
    filtered signal.

    When the macro override forced the filtered `signal` to neutral, an
    actionable "BUY BIAS" / "SELL BIAS" derived from the raw score would be a
    mixed message — downgrade `display_signal` to "NEUTRAL / HOLD".
    `trend` and `confidence_percent` are left untouched (factual gauge
    readings). Never raises; returns a new dict.
    """
    try:
        out = dict(normalized) if isinstance(normalized, dict) else dict(NEUTRAL_DEFAULTS)
        if (bool(macro_override_applied)
                and str(final_signal).lower() == "neutral"
                and out.get("display_signal") != SIGNAL_HOLD):
            out["display_signal"] = SIGNAL_HOLD
        return out
    except Exception:
        return dict(NEUTRAL_DEFAULTS)
