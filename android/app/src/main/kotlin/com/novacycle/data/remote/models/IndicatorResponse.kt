package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Full set of technical indicators from /indicators endpoint.
 * Each field maps 1-to-1 from the backend calculation.
 */
@JsonClass(generateAdapter = true)
data class IndicatorResponse(
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "status") val status: String = "ok",
    @Json(name = "computed_at") val computedAt: String? = null,
    @Json(name = "rsi") val rsi: Float,
    @Json(name = "stoch_k") val stochK: Float,
    @Json(name = "stoch_d") val stochD: Float,
    @Json(name = "stoch_rsi_k") val stochRsiK: Float,
    @Json(name = "stoch_rsi_d") val stochRsiD: Float,
    @Json(name = "macd_line") val macdLine: Float,
    @Json(name = "macd_signal") val macdSignal: Float,
    @Json(name = "macd_histogram") val macdHistogram: Float,
    @Json(name = "sma20") val sma20: Float,
    @Json(name = "sma50") val sma50: Float,
    @Json(name = "sma200") val sma200: Float,
    @Json(name = "bollinger_upper") val bollingerUpper: Float,
    @Json(name = "bollinger_lower") val bollingerLower: Float,
    @Json(name = "bollinger_perc_b") val bollingerPercB: Float,
    @Json(name = "cci") val cci: Float,
    @Json(name = "williams_r") val williamsR: Float,
    @Json(name = "atr") val atr: Float,
    @Json(name = "adx") val adx: Float,
    /** "low", "normal", "high", "extreme", or null when VIX data is missing. */
    @Json(name = "vix_regime") val vixRegime: String?
)
