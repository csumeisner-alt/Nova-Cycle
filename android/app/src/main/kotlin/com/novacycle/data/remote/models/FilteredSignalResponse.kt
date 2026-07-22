package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Pre-filtered signal from /filtered_signal_history.
 * These are already deduplicated by the backend's strongest-confidence rule.
 * The Android client can further apply its own filtering via ApplyFilteredSignalsUseCase.
 */
@JsonClass(generateAdapter = true)
data class FilteredSignalResponse(
    @Json(name = "id") val id: String,
    @Json(name = "timestamp") val timestamp: String,
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "signal_type") val signalType: String,
    @Json(name = "gauge_type") val gaugeType: String,
    @Json(name = "confidence") val confidence: Float,
    @Json(name = "cycle_id") val cycleId: String? = null,
    @Json(name = "session_type") val sessionType: String = "regular"
)
