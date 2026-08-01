package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Request body for POST /register_device.
 * Sent whenever the app obtains a new (or existing) FCM token, and also whenever
 * the user changes notification-related sensitivity settings so the backend always
 * has current preferences before the next signal fires.
 *
 * Preference fields are derived from [com.novacycle.domain.model.SensitivitySettings]:
 *  - minBuyThreshold  — effective minimum confidence (0.0–1.0) for BUY notifications
 *  - minSellThreshold — effective minimum confidence (0.0–1.0) for SELL notifications
 *  - extendedHoursNotifications — whether to receive extended-hours signal alerts
 */
@JsonClass(generateAdapter = true)
data class RegisterDeviceRequest(
    @Json(name = "token") val token: String,
    @Json(name = "device_name") val deviceName: String? = null,
    @Json(name = "min_buy_threshold") val minBuyThreshold: Double = 0.70,
    @Json(name = "min_sell_threshold") val minSellThreshold: Double = 0.70,
    @Json(name = "extended_hours_notifications") val extendedHoursNotifications: Boolean = true,
    /** When true, the backend only notifies this device for high-conviction signals. */
    @Json(name = "high_conviction_only") val highConvictionOnly: Boolean = false
)
