package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Root response from GET /tier_track_record?ticker=VOO&window=90d.
 *
 * Realized historical performance per conviction tier, computed from
 * completed BUY→SELL cycles. Tiers with fewer than [minSampleSize] completed
 * trades report null winRate/avgReturnPercent with sufficientSample=false so
 * the UI shows "not enough signals yet" instead of a misleading percentage.
 *
 * Every field is nullable-safe with defaults so a partial or empty backend
 * response never crashes Moshi parsing — see
 * .agents/memory/android-api-model-sync.md.
 */
@JsonClass(generateAdapter = true)
data class TierTrackRecordResponse(
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "window") val window: String = "90d",
    @Json(name = "available_windows") val availableWindows: List<String> = listOf("30d", "90d", "all"),
    @Json(name = "overall") val overall: TierStats = TierStats(),
    @Json(name = "tiers") val tiers: Map<String, TierStats> = emptyMap(),
    @Json(name = "excluded_price_data_absent") val excludedPriceDataAbsent: Int = 0,
    @Json(name = "min_sample_size") val minSampleSize: Int = 5
) {
    val highConviction: TierStats get() = tiers["high_conviction"] ?: TierStats()
    val opportunity: TierStats get() = tiers["opportunity"] ?: TierStats()
}

@JsonClass(generateAdapter = true)
data class TierStats(
    @Json(name = "trade_count") val tradeCount: Int = 0,
    /** Null when the sample is too small for a reliable percentage */
    @Json(name = "win_rate") val winRate: Float? = null,
    @Json(name = "avg_return_percent") val avgReturnPercent: Float? = null,
    @Json(name = "sufficient_sample") val sufficientSample: Boolean = false
)
