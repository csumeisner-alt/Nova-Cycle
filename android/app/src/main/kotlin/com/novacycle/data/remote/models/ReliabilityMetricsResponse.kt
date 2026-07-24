package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Aggregate reliability metrics computed from the BUY→SELL trade cycles.
 * Mirrors the backend reliability_engine.compute_metrics output.
 */
@JsonClass(generateAdapter = true)
data class ReliabilityMetricsResponse(
    @Json(name = "win_rate") val winRate: Float = 0f,
    @Json(name = "average_return_percent") val averageReturnPercent: Float = 0f,
    @Json(name = "median_return_percent") val medianReturnPercent: Float = 0f,
    @Json(name = "average_return_dollars") val averageReturnDollars: Float = 0f,
    @Json(name = "median_return_dollars") val medianReturnDollars: Float = 0f,
    @Json(name = "average_hold_time") val averageHoldTime: Float = 0f,
    @Json(name = "best_trade") val bestTrade: TradeCycleResponse? = null,
    @Json(name = "worst_trade") val worstTrade: TradeCycleResponse? = null,
    @Json(name = "return_distribution") val returnDistribution: List<ReturnDistributionBin> = emptyList(),
    @Json(name = "reliability_by_volatility_class") val reliabilityByVolatilityClass: Map<String, SegmentMetrics> = emptyMap(),
    @Json(name = "reliability_by_liquidity_class") val reliabilityByLiquidityClass: Map<String, SegmentMetrics> = emptyMap(),
    @Json(name = "reliability_by_session_type") val reliabilityBySessionType: Map<String, SegmentMetrics> = emptyMap()
)

@JsonClass(generateAdapter = true)
data class ReturnDistributionBin(
    @Json(name = "min") val min: Float,
    @Json(name = "max") val max: Float,
    @Json(name = "count") val count: Int
)

@JsonClass(generateAdapter = true)
data class SegmentMetrics(
    @Json(name = "count") val count: Int = 0,
    @Json(name = "win_rate") val winRate: Float = 0f,
    @Json(name = "average_return_percent") val averageReturnPercent: Float = 0f,
    @Json(name = "median_return_percent") val medianReturnPercent: Float = 0f
)
