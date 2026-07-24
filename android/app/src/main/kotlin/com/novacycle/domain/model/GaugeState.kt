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
    val isLoading: Boolean = false
) {
    val isBuy: Boolean get() = signal.lowercase() == "buy"
    val isSell: Boolean get() = signal.lowercase() == "sell"

    /** Normalized 0–1 position for the gauge needle arc (0=full sell, 1=full buy) */
    val normalizedPosition: Float get() = (score + 100f) / 200f
}
