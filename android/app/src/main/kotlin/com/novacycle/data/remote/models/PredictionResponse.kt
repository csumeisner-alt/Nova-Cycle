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
    /** Normalized confidence 0–100 (integer, sigmoid-clamped by the backend). */
    @Json(name = "confidence_percent") val confidencePercent: Int = 0,
    /** "UP", "DOWN", or "NEUTRAL" — trend direction from the raw score sign. */
    @Json(name = "trend") val trend: String = "NEUTRAL",
    /** "BUY BIAS", "SELL BIAS", or "NEUTRAL / HOLD" (65% confidence rule). */
    @Json(name = "display_signal") val displaySignal: String = "NEUTRAL / HOLD",
    /** Per-indicator contribution scores (values may be numeric or textual annotations). */
    @Json(name = "indicator_breakdown") val indicatorBreakdown: Map<String, Any> = emptyMap(),
    @Json(name = "ml_confidence") val mlConfidence: Float = 0f,
    @Json(name = "ml_fallback") val mlFallback: Boolean = false,
    @Json(name = "liquidity_score") val liquidityScore: Float = 0f,
    @Json(name = "gap_type") val gapType: String? = null,
    @Json(name = "gap_momentum") val gapMomentum: Float? = null,
    @Json(name = "macro_override_applied") val macroOverrideApplied: Boolean = false,
    @Json(name = "macro_override_reason") val macroOverrideReason: String? = null,
    @Json(name = "decision_filter_applied") val decisionFilterApplied: Boolean = false,
    @Json(name = "decision_filter_reason") val decisionFilterReason: String? = null,
    @Json(name = "cycle_quality_score") val cycleQualityScore: Float = 0f,
    @Json(name = "volatility_regime") val volatilityRegime: String? = null,
    @Json(name = "liquidity_class") val liquidityClass: String? = null,
    @Json(name = "confidence_momentum") val confidenceMomentum: Float = 0f,
    /** "opportunity" | "high_conviction" | null for neutral signals. */
    @Json(name = "conviction_tier") val convictionTier: String? = null,
    /** Plain-language explanation of the conviction tier decision. */
    @Json(name = "conviction_reasons") val convictionReasons: List<String> = emptyList(),
    @Json(name = "session_type") val sessionType: String? = null,
    @Json(name = "is_extended_hours") val isExtendedHours: Boolean = false,
    @Json(name = "note") val note: String? = null,
    @Json(name = "timestamp") val timestamp: String = "",
    @Json(name = "ticker") val ticker: String = "VOO"
)
