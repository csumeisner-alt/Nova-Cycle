package com.novacycle.domain.model

/**
 * Represents the current state of one gauge (long or short).
 * Drives the animated DualGaugeWidget composable.
 */
data class GaugeState(
    /** Score from -100 to +100 driving needle position */
    val score: Float = 0f,
    /** "buy", "sell", or "neutral" */
    val signal: String = "neutral",
    /** 0–100 confidence percentage */
    val confidence: Float = 0f,
    /** "long" or "short" */
    val gaugeType: String = "long",
    val ticker: String = "VOO",
    val isLoading: Boolean = false,
    /** Normalized confidence 0–100 from the backend (clamped app-side too) */
    val confidencePercent: Int = 0,
    /** "UP", "DOWN", or "NEUTRAL" */
    val trend: String = "NEUTRAL",
    /** "BUY BIAS", "SELL BIAS", or "NEUTRAL / HOLD" */
    val displaySignal: String = "NEUTRAL / HOLD",
    /** True when showing the no-data fallback — gauge renders gray */
    val isFallback: Boolean = false,
    /** "opportunity" | "high_conviction" | null when the signal is neutral */
    val convictionTier: String? = null,
    /** Directional gauge position: raw score -100..100 mapped to 0..100. */
    val gaugePercent: Int = ((score + 100f) / 2f).toInt().coerceIn(0, 100)
) {
    val isBuy: Boolean get() = signal.lowercase() == "buy"
    val isSell: Boolean get() = signal.lowercase() == "sell"

    /** Normalized 0–1 position for the gauge needle arc (0=full sell, 1=full buy) */
    val normalizedPosition: Float get() = (score + 100f) / 200f

    /** Confidence zone for the normalized 0–100% confidence display */
    val confidenceZone: ConfidenceZone get() = ConfidenceZone.fromPercent(confidencePercent)

    val gaugeZone: ConfidenceZone get() = ConfidenceZone.fromPercent(gaugePercent)

    val gaugeAction: String
        get() = when {
            gaugePercent >= 65 -> "BUY"
            gaugePercent <= 35 -> "SELL"
            else -> "HOLD"
        }
}

/**
 * Display zones for the normalized confidence percentage:
 *   0–30  → WEAK (red), 31–64 → UNCERTAIN (yellow), 65–100 → STRONG (green).
 */
enum class ConfidenceZone(val label: String) {
    WEAK("Weak"),
    UNCERTAIN("Uncertain"),
    STRONG("Strong");

    companion object {
        fun fromPercent(percent: Int): ConfidenceZone {
            val p = percent.coerceIn(0, 100)
            return when {
                p <= 30 -> WEAK
                p <= 64 -> UNCERTAIN
                else -> STRONG
            }
        }
    }
}
