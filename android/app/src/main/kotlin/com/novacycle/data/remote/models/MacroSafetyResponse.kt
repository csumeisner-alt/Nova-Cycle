package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Response from GET /api/macro_safety.
 *
 * Field names mirror the backend JSON exactly (snake_case via @Json) —
 * the backend shape is the source of truth; any change there must be
 * mirrored here or Moshi parsing crashes at launch.
 */
@JsonClass(generateAdapter = true)
data class MacroSafetyResponse(
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "status") val status: String = "ok",
    @Json(name = "computed_at") val computedAt: String? = null,
    @Json(name = "vix_close") val vixClose: Float? = null,
    @Json(name = "vix_regime") val vixRegime: String? = null,
    @Json(name = "vix_timestamp") val vixTimestamp: String? = null,
    @Json(name = "long_score") val longScore: Float = 0f,
    @Json(name = "override_active") val overrideActive: Boolean = false,
    @Json(name = "suppresses_short_buy") val suppressesShortBuy: Boolean = false,
    @Json(name = "suppresses_short_sell") val suppressesShortSell: Boolean = false,
    @Json(name = "reason") val reason: String = "",
    @Json(name = "thresholds") val thresholds: MacroSafetyThresholds? = null,
    @Json(name = "last_override_applied_at") val lastOverrideAppliedAt: String? = null,
    /** True when the database has no VIX rows at all. */
    @Json(name = "vix_data_missing") val vixDataMissing: Boolean = false,
    /** True when VIX is behind VOO by more than the allowed trading-day lag. */
    @Json(name = "vix_is_stale") val vixIsStale: Boolean = false,
    /** Raw elapsed age for diagnostics; weekends and holidays are included. */
    @Json(name = "vix_staleness_hours") val vixStalenessHours: Float? = null,
    /** Number of VOO trading days by which VIX trails, or null when unavailable. */
    @Json(name = "vix_trading_day_lag") val vixTradingDayLag: Int? = null
)

@JsonClass(generateAdapter = true)
data class MacroSafetyThresholds(
    @Json(name = "long_strong_bear") val longStrongBear: Float = -70f,
    @Json(name = "long_strong_bull") val longStrongBull: Float = 70f,
    @Json(name = "ml_override_threshold") val mlOverrideThreshold: Float = 0.80f
)
