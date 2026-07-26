package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Request body for POST /billing/verify_purchase.
 * Sends the opaque Play purchase token to the backend, which verifies it
 * against the Google Play Developer API before the app unlocks Mint Luxe.
 */
@JsonClass(generateAdapter = true)
data class VerifyPurchaseRequest(
    @Json(name = "product_id") val productId: String,
    @Json(name = "purchase_token") val purchaseToken: String
)

/**
 * Response from /billing/verify_purchase and /billing/entitlement.
 * `state` is one of: active, revoked, pending, invalid.
 */
@JsonClass(generateAdapter = true)
data class EntitlementResponse(
    @Json(name = "entitled") val entitled: Boolean,
    @Json(name = "state") val state: String,
    @Json(name = "product_id") val productId: String
)
