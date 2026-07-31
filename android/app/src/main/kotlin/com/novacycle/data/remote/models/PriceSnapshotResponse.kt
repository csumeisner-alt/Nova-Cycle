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
    @Json(name = "long_model_price") val longModelPrice: Float? = null,
    @Json(name = "long_model_timestamp") val longModelTimestamp: String? = null,
    @Json(name = "short_model_price") val shortModelPrice: Float? = null,
    @Json(name = "short_model_timestamp") val shortModelTimestamp: String? = null
)