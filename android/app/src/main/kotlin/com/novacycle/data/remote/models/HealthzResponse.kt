package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Response from GET /healthz.
 *
 * Mirrors the backend health payload: overall status is "ok" or "degraded",
 * `models` reports per-model training/fallback health, and `alerts` carries
 * human-readable degradation messages (same ones the web status page shows).
 * Unknown/extra fields are ignored by Moshi.
 */
@JsonClass(generateAdapter = true)
data class HealthzResponse(
    @Json(name = "status") val status: String,
    @Json(name = "service") val service: String? = null,
    @Json(name = "timestamp") val timestamp: String? = null,
    @Json(name = "models") val models: Map<String, ModelHealth>? = null,
    @Json(name = "alerts") val alerts: List<String>? = null
) {
    val isDegraded: Boolean get() = status == "degraded"

    /** Names of models that are unavailable or failed their last training (same rule as the web banner). */
    val degradedModels: List<String>
        get() = models.orEmpty()
            .filter { (_, m) -> m.neutralFallback == true || m.lastTrainingSuccess == false }
            .keys
            .toList()
}

@JsonClass(generateAdapter = true)
data class ModelHealth(
    @Json(name = "last_training_success") val lastTrainingSuccess: Boolean? = null,
    @Json(name = "last_training_error") val lastTrainingError: String? = null,
    @Json(name = "last_trained_at") val lastTrainedAt: String? = null,
    @Json(name = "neutral_fallback") val neutralFallback: Boolean? = null
)
