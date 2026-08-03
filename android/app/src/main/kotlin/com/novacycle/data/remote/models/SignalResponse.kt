package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * A single trading signal event from /signal_history.
 */
@JsonClass(generateAdapter = true)
data class SignalResponse(
    @Json(name = "id") val id: String,
    @Json(name = "timestamp") val timestamp: String,
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "cycle_id") val cycleId: String? = null,
    /** "buy" or "sell" */
    @Json(name = "signal_type") val signalType: String,
    /** "long" or "short" */
    @Json(name = "gauge_type") val gaugeType: String,
    @Json(name = "confidence") val confidence: Float,
    @Json(name = "session_type") val sessionType: String = "regular",
    @Json(name = "is_extended_hours") val isExtendedHours: Boolean = false,
    @Json(name = "gap_type") val gapType: String? = null,
    @Json(name = "liquidity_score") val liquidityScore: Float = 1f,
    @Json(name = "macro_override_applied") val macroOverrideApplied: Boolean = false,
    /** "opportunity" | "high_conviction" | null for pre-tiering rows. */
    @Json(name = "conviction_tier") val convictionTier: String? = null,
    @Json(name = "conviction_reasons") val convictionReasons: List<String> = emptyList(),
    /**
     * Model reliability when the signal was stored:
     * "healthy" | "model_unavailable" | "training_stuck" | "stale_rolled_back".
     * Null for rows recorded before this field existed (treated as unknown, not degraded).
     */
    @Json(name = "model_state") val modelState: String? = null
)
