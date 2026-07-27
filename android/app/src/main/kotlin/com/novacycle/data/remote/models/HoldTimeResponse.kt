package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Estimated position hold duration from /hold_time_estimate.
 */
@JsonClass(generateAdapter = true)
data class HoldTimeResponse(
    @Json(name = "minutes") val minutes: Int,
    @Json(name = "human_readable") val humanReadable: String,
    @Json(name = "confidence") val confidence: Float,
    @Json(name = "reasoning") val reasoning: List<String> = emptyList(),
    @Json(name = "timestamp") val timestamp: String? = null,
    @Json(name = "ticker") val ticker: String = "VOO"
)
