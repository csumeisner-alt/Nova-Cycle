package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Response from /predict_long and /predict_short.
 * The score drives the gauge needle position (−100 to +100 scale expected).
 */
@JsonClass(generateAdapter = true)
data class PredictionResponse(
    @Json(name = "score") val score: Float,
    /** "buy", "sell", or "neutral" */
    @Json(name = "signal") val signal: String,
    @Json(name = "confidence") val confidence: Float,
    /** Normalized confidence 0–100 (integer, sigmoid-clamped by the backend). */
    @Json(name = "confidence_percent") val confidencePercent: Int = 0,
    /** "UP", "DOWN", or "NEUTRAL" — trend direction from the raw score sign. */
    @Json(name = "trend") val trend: String = "NEUTRAL",
    /** "BUY BIAS", "SELL BIAS", or "NEUTRAL / HOLD" (65% confidence rule). */
    @Json(name = "display_signal") val displaySignal: String = "NEUTRAL / HOLD",
    /** Per-indicator contribution scores (values may be numeric or textual annotations). */
    @Json(name = "indicator_breakdown") val indicatorBreakdown: Map<String, Any> = emptyMap(),
    @Json(name = "ml_confidence") val mlConfidence: Float = 0f,
    @Json(name = "ml_fallback") val mlFallback: Boolean = false,
    @Json(name = "liquidity_score") val liquidityScore: Float = 0f,
    @Json(name = "gap_type") val gapType: String? = null,
    @Json(name = "gap_momentum") val gapMomentum: Float? = null,
    @Json(name = "macro_override_applied") val macroOverrideApplied: Boolean = false,
    @Json(name = "macro_override_reason") val macroOverrideReason: String? = null,
    @Json(name = "decision_filter_applied") val decisionFilterApplied: Boolean = false,
    @Json(name = "decision_filter_reason") val decisionFilterReason: String? = null,
    @Json(name = "cycle_quality_score") val cycleQualityScore: Float = 0f,
    @Json(name = "volatility_regime") val volatilityRegime: String? = null,
    @Json(name = "liquidity_class") val liquidityClass: String? = null,
    @Json(name = "confidence_momentum") val confidenceMomentum: Float = 0f,
    /** "opportunity" | "high_conviction" | null for neutral signals. */
    @Json(name = "conviction_tier") val convictionTier: String? = null,
    /** Plain-language explanation of the conviction tier decision. */
    @Json(name = "conviction_reasons") val convictionReasons: List<String> = emptyList(),
    /**
     * True when the decision filter blocked this signal as a soft (non-safety-critical) block.
     * The raw gauge direction is available as [candidateSignal]. Candidates are never stored in
     * signal history and never trigger push notifications — they are informational only.
     */
    @Json(name = "is_candidate") val isCandidate: Boolean = false,
    /**
     * The raw gauge direction ("buy" or "sell") when [isCandidate] is true; null otherwise.
     * Use this to display the underlying pressure even when the signal is not executable.
     */
    @Json(name = "candidate_signal") val candidateSignal: String? = null,
    /** "opportunity" | null — conviction tier for the candidate direction (informational only). */
    @Json(name = "candidate_conviction_tier") val candidateConvictionTier: String? = null,
    /** Plain-language explanation of why the candidate was not promoted to executable. */
    @Json(name = "candidate_conviction_reasons") val candidateConvictionReasons: List<String> = emptyList(),
    @Json(name = "session_type") val sessionType: String? = null,
    @Json(name = "is_extended_hours") val isExtendedHours: Boolean = false,
    @Json(name = "note") val note: String? = null,
    @Json(name = "timestamp") val timestamp: String = "",
    @Json(name = "ticker") val ticker: String = "VOO",
    /**
     * Explicit model availability semantics. Values:
     *   "healthy"          — model trained and prediction is reliable.
     *   "model_unavailable"— model file missing or failed to load; output is
     *                        a neutral 0.5 fallback.
     *   "training_stuck"   — repeated consecutive retraining failures; running
     *                        on a stale model checkpoint.
     *   "stale_rolled_back"— last retrain regressed and was rolled back; model
     *                        is older than the most recent training attempt.
     *   "baseline_mode"    — no trained model passes the OOS quality gate;
     *                        long signal uses the calibrated majority-class base
     *                        rate (~73% bull bias) instead of a trained model.
     *                        predictionReliable is always false in this state.
     * Null when the field was not present in an older backend response.
     */
    @Json(name = "model_state") val modelState: String? = null,
    /** "trained" when a gate-passing model is active; "baseline" when the long
     *  signal falls back to the calibrated majority-class base rate. Absent on
     *  short-trend responses and older backend versions. */
    @Json(name = "long_signal_mode") val longSignalMode: String? = null,
    /**
     * False when the prediction should be presented as degraded rather than as
     * a normal signal (i.e. when model_state is anything other than "healthy").
     * Defaults to true so older backend responses without this field are
     * treated as reliable.
     */
    @Json(name = "prediction_reliable") val predictionReliable: Boolean = true
)
