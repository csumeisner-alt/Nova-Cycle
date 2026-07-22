package com.novacycle.domain.model

/**
 * Domain model for a single confidence data point used in charts.
 * momentum = confidence(t) - confidence(t-1), computed in ViewModel.
 */
data class ConfidencePoint(
    val timestamp: String,
    val ticker: String = "VOO",
    val longBuyConfidence: Float,
    val longSellConfidence: Float,
    val shortBuyConfidence: Float,
    val shortSellConfidence: Float,
    val isExtendedHours: Boolean = false,
    /** Rate of change since previous point — computed client-side */
    val longMomentum: Float = 0f,
    val shortMomentum: Float = 0f
)
