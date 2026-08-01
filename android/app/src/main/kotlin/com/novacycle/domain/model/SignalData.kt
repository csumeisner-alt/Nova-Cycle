package com.novacycle.domain.model

/**
 * Domain model for a trading signal — maps from both SignalResponse and FilteredSignalResponse.
 * Used throughout the UI layer to decouple from remote DTO shapes.
 */
data class SignalData(
    val id: String,
    val timestamp: String,
    val ticker: String = "VOO",
    val cycleId: String? = null,
    /** "buy" or "sell" */
    val signalType: String,
    /** "long" or "short" */
    val gaugeType: String,
    /** Normalized confidence in the backend's 0–1 representation. */
    val confidence: Float,
    val sessionType: String = "regular",
    val isExtendedHours: Boolean = false,
    val gapType: String? = null,
    val liquidityScore: Float = 1f,
    val macroOverrideApplied: Boolean = false,
    /** "opportunity" | "high_conviction" | null for pre-tiering signals. */
    val convictionTier: String? = null,
    val convictionReasons: List<String> = emptyList(),
    /**
     * True when the decision filter soft-blocked this signal as a candidate.
     * The signal direction is real but not yet executable — display as an
     * informational hint rather than an actionable trade alert.
     * Candidates are never stored in signal history and never push-notify.
     */
    val isCandidate: Boolean = false
) {
    val isBuy: Boolean get() = signalType.lowercase() == "buy"
    val isSell: Boolean get() = signalType.lowercase() == "sell"
    val isLongGauge: Boolean get() = gaugeType.lowercase() == "long"
    val isHighConviction: Boolean get() = convictionTier == "high_conviction"
    val isOpportunity: Boolean get() = convictionTier == "opportunity" && !isCandidate
}
