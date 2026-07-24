package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Root response from GET /trade_history?ticker=VOO.
 * Contains the list of completed BUY→SELL trade cycles plus summary metrics.
 */
@JsonClass(generateAdapter = true)
data class TradeHistoryResponse(
    @Json(name = "ticker") val ticker: String,
    @Json(name = "cycles") val cycles: List<TradeCycleResponse> = emptyList(),
    @Json(name = "summary") val summary: ReliabilityMetricsResponse
)
