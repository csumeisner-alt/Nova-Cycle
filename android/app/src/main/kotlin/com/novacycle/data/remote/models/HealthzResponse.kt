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
    @Json(name = "ticker") val ticker: String? = null,
    @Json(name = "timestamp") val timestamp: String? = null,
    @Json(name = "models") val models: Map<String, ModelHealth>? = null,
    @Json(name = "alerts") val alerts: List<String>? = null,
    @Json(name = "spx_futures") val spxFutures: Map<String, Any?>? = null,
    @Json(name = "vix") val vix: Map<String, Any?>? = null,
    @Json(name = "voo_5min") val voo5min: Map<String, Any?>? = null,
    @Json(name = "voo_5min_recovery") val voo5minRecovery: Map<String, Any?>? = null
) {
    val isDegraded: Boolean get() = status == "degraded"

    /** Names of models that are unavailable or failed their last training (same rule as the web banner). */
    val degradedModels: List<String>
        get() = models.orEmpty()
            .filter { (_, m) -> m.neutralFallback == true || m.lastTrainingSuccess == false || m.trainingStuck == true }
            .keys
            .toList()
}

@JsonClass(generateAdapter = true)
data class ModelHealth(
    @Json(name = "last_training_success") val lastTrainingSuccess: Boolean? = null,
    @Json(name = "last_retrain_outcome") val lastRetrainOutcome: String? = null,
    @Json(name = "last_retrain_rolled_back") val lastRetrainRolledBack: Boolean? = null,
    @Json(name = "last_retrain_attempted_accuracy") val lastRetrainAttemptedAccuracy: Float? = null,
    @Json(name = "active_model_accuracy") val activeModelAccuracy: Float? = null,
    @Json(name = "consecutive_training_failures") val consecutiveTrainingFailures: Int? = null,
    @Json(name = "training_stuck") val trainingStuck: Boolean? = null,
    @Json(name = "last_training_error") val lastTrainingError: String? = null,
    @Json(name = "last_training_attempted_at") val lastTrainingAttemptedAt: String? = null,
    @Json(name = "last_training_accuracy") val lastTrainingAccuracy: Float? = null,
    @Json(name = "last_trained_at") val lastTrainedAt: String? = null,
    @Json(name = "neutral_fallback") val neutralFallback: Boolean? = null,
    @Json(name = "ml_fallback_count") val mlFallbackCount: Int? = null,
    @Json(name = "ml_fallback_last_at") val mlFallbackLastAt: String? = null,
    @Json(name = "ml_fallback_last_reason") val mlFallbackLastReason: String? = null,
    @Json(name = "ml_fallback_total_count") val mlFallbackTotalCount: Int? = null,
    @Json(name = "ml_fallback_total_last_at") val mlFallbackTotalLastAt: String? = null,
    @Json(name = "ml_fallback_total_last_reason") val mlFallbackTotalLastReason: String? = null
)
