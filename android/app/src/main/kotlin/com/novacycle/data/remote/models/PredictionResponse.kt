package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Response from /predict_long and /predict_short.
 * The score drives the gauge needle position (−100 to +100 scale expected).
 */
@JsonClass(generateAdapter = true)
data class PredictionResponse(
    @Json(name = "score") val score: Float,
    /** "buy", "sell", or "neutral" */
    @Json(name = "signal") val signal: String,
    @Json(name = "confidence") val confidence: Float,
    /** Per-indicator contribution scores */
    @Json(name = "indicator_breakdown") val indicatorBreakdown: Map<String, Float> = emptyMap(),
    @Json(name = "ml_confidence") val mlConfidence: Float = 0f,
    @Json(name = "liquidity_score") val liquidityScore: Float = 0f,
    @Json(name = "gap_type") val gapType: String? = null,
    @Json(name = "macro_override_applied") val macroOverrideApplied: Boolean = false,
    @Json(name = "timestamp") val timestamp: String = "",
    @Json(name = "ticker") val ticker: String = "VOO"
)
