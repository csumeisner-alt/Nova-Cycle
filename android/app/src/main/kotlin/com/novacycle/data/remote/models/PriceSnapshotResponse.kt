package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Prices shown on the charts, sourced from the same candle stream used by
 * the prediction endpoints.
 */
@JsonClass(generateAdapter = true)
data class PriceSnapshotResponse(
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "current_price") val currentPrice: Float? = null,
    @Json(name = "current_timestamp") val currentTimestamp: String? = null,
    @Json(name = "current_session") val currentSession: String? = null,
    @Json(name = "is_extended_hours") val isExtendedHours: Boolean = false,
    @Json(name = "reference_price") val referencePrice: Float? = null,
    @Json(name = "reference_timestamp") val referenceTimestamp: String? = null,
    @Json(name = "day_change_percent") val dayChangePercent: Float? = null,
    @Json(name = "day_direction") val dayDirection: String? = null,
    @Json(name = "long_model_price") val longModelPrice: Float? = null,
    @Json(name = "long_model_timestamp") val longModelTimestamp: String? = null,
    @Json(name = "short_model_price") val shortModelPrice: Float? = null,
    @Json(name = "short_model_timestamp") val shortModelTimestamp: String? = null
)