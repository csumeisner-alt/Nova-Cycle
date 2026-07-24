package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Request body for POST /register_device.
 * Sent whenever the app obtains a new (or existing) FCM token.
 */
@JsonClass(generateAdapter = true)
data class RegisterDeviceRequest(
    @Json(name = "token") val token: String,
    @Json(name = "device_name") val deviceName: String? = null
)
