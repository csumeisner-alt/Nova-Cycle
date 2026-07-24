package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * One confidence snapshot from /confidence_history.
 * Contains separate confidence values for long and short gauges, buy and sell sides.
 */
@JsonClass(generateAdapter = true)
data class ConfidenceHistoryResponse(
    @Json(name = "id") val id: String,
    @Json(name = "timestamp") val timestamp: String,
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "long_buy_confidence") val longBuyConfidence: Float,
    @Json(name = "long_sell_confidence") val longSellConfidence: Float,
    @Json(name = "short_buy_confidence") val shortBuyConfidence: Float,
    @Json(name = "short_sell_confidence") val shortSellConfidence: Float,
    @Json(name = "session_type") val sessionType: String = "regular",
    @Json(name = "is_extended_hours") val isExtendedHours: Boolean = false
)
