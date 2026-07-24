package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * A single completed BUY→SELL trade cycle.
 * All monetary/percent fields are nullable so the backend can evolve without
 * breaking the client when no cycles have completed yet.
 */
@JsonClass(generateAdapter = true)
data class TradeCycleResponse(
    @Json(name = "cycle_id") val cycleId: String,
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "buy_timestamp") val buyTimestamp: String? = null,
    @Json(name = "sell_timestamp") val sellTimestamp: String? = null,
    @Json(name = "buy_price") val buyPrice: Float? = null,
    @Json(name = "sell_price") val sellPrice: Float? = null,
    @Json(name = "return_percent") val returnPercent: Float? = null,
    @Json(name = "return_dollars") val returnDollars: Float? = null,
    @Json(name = "hold_time_minutes") val holdTimeMinutes: Float? = null,
    @Json(name = "confidence_at_buy") val confidenceAtBuy: Float? = null,
    @Json(name = "confidence_at_sell") val confidenceAtSell: Float? = null,
    @Json(name = "session_type_at_buy") val sessionTypeAtBuy: String? = null,
    @Json(name = "liquidity_score_at_buy") val liquidityScoreAtBuy: Float? = null,
    @Json(name = "gap_type_at_buy") val gapTypeAtBuy: String? = null,
    @Json(name = "macro_override_applied") val macroOverrideApplied: Boolean = false,
    @Json(name = "volatility_class") val volatilityClass: String? = null,
    @Json(name = "liquidity_class") val liquidityClass: String? = null
)
