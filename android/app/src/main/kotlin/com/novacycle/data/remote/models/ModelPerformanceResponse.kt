package com.novacycle.data.remote.models

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass

/**
 * Root response from GET /model_performance?ticker=VOO.
 *
 * Every field is nullable-safe (with sane defaults) so a partial or empty
 * backend response never crashes Moshi parsing — see
 * .agents/memory/android-api-model-sync.md. On an empty DB the backend returns
 * 200 with zeroed summary, empty arrays and null best/worst trades.
 */
@JsonClass(generateAdapter = true)
data class ModelPerformanceResponse(
    @Json(name = "ticker") val ticker: String = "VOO",
    @Json(name = "period") val period: String = "day",
    @Json(name = "window") val window: String = "90d",
    @Json(name = "summary") val summary: ModelPerformanceSummary = ModelPerformanceSummary(),
    @Json(name = "periods") val periods: List<PerformancePeriod> = emptyList(),
    @Json(name = "confidence_buckets") val confidenceBuckets: Map<String, ConfidenceBucket> = emptyMap(),
    @Json(name = "calibration_curve") val calibrationCurve: List<CalibrationPoint> = emptyList(),
    @Json(name = "cumulative_pnl") val cumulativePnl: List<CumulativePnlPoint> = emptyList(),
    @Json(name = "return_distribution") val returnDistribution: List<ReturnDistributionLabelBin> = emptyList(),
    @Json(name = "session_breakdown") val sessionBreakdown: Map<String, PerformanceSegment> = emptyMap(),
    @Json(name = "vix_regime_breakdown") val vixRegimeBreakdown: Map<String, PerformanceSegment> = emptyMap(),
    @Json(name = "best_trade") val bestTrade: TradeCycleResponse? = null,
    @Json(name = "worst_trade") val worstTrade: TradeCycleResponse? = null,
    @Json(name = "streak") val streak: StreakInfo = StreakInfo(),
    @Json(name = "missed_rallies") val missedRallies: MissedRallies = MissedRallies(),
    @Json(name = "accuracy_history") val accuracyHistory: List<AccuracyHistoryEntry> = emptyList()
)

@JsonClass(generateAdapter = true)
data class ModelPerformanceSummary(
    @Json(name = "total_trades") val totalTrades: Int = 0,
    @Json(name = "wins") val wins: Int = 0,
    @Json(name = "losses") val losses: Int = 0,
    @Json(name = "buy_precision") val buyPrecision: Float = 0f,
    @Json(name = "avg_return_percent") val avgReturnPercent: Float = 0f,
    @Json(name = "missed_rally_rate") val missedRallyRate: Float = 0f,
    @Json(name = "current_win_streak") val currentWinStreak: Int = 0,
    @Json(name = "recommendation_stability") val recommendationStability: Float = 0f,
    @Json(name = "avg_confidence") val avgConfidence: Float = 0f,
    @Json(name = "cumulative_return_percent") val cumulativeReturnPercent: Float = 0f
)

@JsonClass(generateAdapter = true)
data class PerformancePeriod(
    @Json(name = "label") val label: String = "",
    @Json(name = "start") val start: String? = null,
    @Json(name = "buy_count") val buyCount: Int = 0,
    @Json(name = "wins") val wins: Int = 0,
    @Json(name = "losses") val losses: Int = 0,
    @Json(name = "precision") val precision: Float = 0f,
    @Json(name = "avg_return_percent") val avgReturnPercent: Float = 0f,
    @Json(name = "missed_rallies") val missedRallies: Int = 0,
    @Json(name = "avg_confidence") val avgConfidence: Float = 0f,
    @Json(name = "oos_accuracy") val oosAccuracy: Float? = null
)

@JsonClass(generateAdapter = true)
data class ConfidenceBucket(
    @Json(name = "trade_count") val tradeCount: Int = 0,
    @Json(name = "win_rate") val winRate: Float = 0f,
    @Json(name = "avg_return_percent") val avgReturnPercent: Float = 0f
)

@JsonClass(generateAdapter = true)
data class CalibrationPoint(
    @Json(name = "confidence_mid") val confidenceMid: Float = 0f,
    @Json(name = "actual_win_rate") val actualWinRate: Float? = null,
    @Json(name = "trade_count") val tradeCount: Int = 0
)

@JsonClass(generateAdapter = true)
data class CumulativePnlPoint(
    @Json(name = "timestamp") val timestamp: String? = null,
    @Json(name = "cumulative_return_percent") val cumulativeReturnPercent: Float = 0f
)

@JsonClass(generateAdapter = true)
data class ReturnDistributionLabelBin(
    @Json(name = "label") val label: String = "",
    @Json(name = "min") val min: Float = 0f,
    @Json(name = "max") val max: Float = 0f,
    @Json(name = "count") val count: Int = 0
)

@JsonClass(generateAdapter = true)
data class PerformanceSegment(
    @Json(name = "count") val count: Int = 0,
    @Json(name = "win_rate") val winRate: Float = 0f,
    @Json(name = "average_return_percent") val averageReturnPercent: Float = 0f
)

@JsonClass(generateAdapter = true)
data class StreakInfo(
    @Json(name = "current_win") val currentWin: Int = 0,
    @Json(name = "current_loss") val currentLoss: Int = 0,
    @Json(name = "longest_win") val longestWin: Int = 0,
    @Json(name = "longest_loss") val longestLoss: Int = 0
)

@JsonClass(generateAdapter = true)
data class MissedRallies(
    @Json(name = "count") val count: Int = 0,
    @Json(name = "timestamps") val timestamps: List<String> = emptyList(),
    @Json(name = "rate") val rate: Float = 0f
)

@JsonClass(generateAdapter = true)
data class AccuracyHistoryEntry(
    @Json(name = "model_name") val modelName: String = "",
    @Json(name = "trained_at") val trainedAt: String? = null,
    @Json(name = "accuracy") val accuracy: Float? = null
)
