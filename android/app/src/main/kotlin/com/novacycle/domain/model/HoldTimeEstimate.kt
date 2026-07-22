package com.novacycle.domain.model

/**
 * Domain model for hold-time prediction result.
 */
data class HoldTimeEstimate(
    val minutes: Int,
    val humanReadable: String,
    val confidence: Float,
    val reasoning: String,
    val ticker: String = "VOO"
) {
    /** Derived hours display for large values */
    val hours: Float get() = minutes / 60f
    val isShortTerm: Boolean get() = minutes <= 60
}
