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
