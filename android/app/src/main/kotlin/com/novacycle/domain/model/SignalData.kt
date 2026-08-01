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
    /** 0–100 confidence percentage */
    val confidence: Float,
    val sessionType: String = "regular",
    val isExtendedHours: Boolean = false,
    val gapType: String? = null,
    val liquidityScore: Float = 1f,
    val macroOverrideApplied: Boolean = false,
    /** "opportunity" | "high_conviction" | null for pre-tiering signals. */
    val convictionTier: String? = null,
    val convictionReasons: List<String> = emptyList()
) {
    val isBuy: Boolean get() = signalType.lowercase() == "buy"
    val isSell: Boolean get() = signalType.lowercase() == "sell"
    val isLongGauge: Boolean get() = gaugeType.lowercase() == "long"
    val isHighConviction: Boolean get() = convictionTier == "high_conviction"
}
