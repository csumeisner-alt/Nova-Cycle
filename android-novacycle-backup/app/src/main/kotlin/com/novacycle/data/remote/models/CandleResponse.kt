package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * OHLCV candlestick data point from /voo_candles.
 * Used as the base for the chart background in Raw and Filtered chart screens.
 */
@JsonClass(generateAdapter = true)
data class CandleResponse(
    @Json(name = "timestamp") val timestamp: String,
    @Json(name = "open") val open: Float,
    @Json(name = "high") val high: Float,
    @Json(name = "low") val low: Float,
    @Json(name = "close") val close: Float,
    @Json(name = "volume") val volume: Long = 0L,
    @Json(name = "is_extended_hours") val isExtendedHours: Boolean = false,
    @Json(name = "session_type") val sessionType: String = "regular",
    /** Percentage gap from prior close (can be null if no gap) */
    @Json(name = "gap_percent") val gapPercent: Float? = null,
    @Json(name = "gap_type") val gapType: String? = null
)
