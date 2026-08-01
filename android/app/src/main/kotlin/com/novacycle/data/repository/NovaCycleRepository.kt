package com.novacycle.data.repository

import com.novacycle.data.local.dao.CandleDao
import com.novacycle.data.local.dao.ConfidenceDao
import com.novacycle.data.local.dao.SignalDao
import com.novacycle.data.local.entities.CandleEntity
import com.novacycle.data.local.entities.ConfidenceHistoryEntity
import com.novacycle.data.local.entities.SignalHistoryEntity
import com.novacycle.data.remote.NovaCycleApiService
import com.novacycle.data.remote.models.*
import com.novacycle.domain.model.NotifSensitivity
import com.novacycle.domain.model.SensitivitySettings
import javax.inject.Inject
import javax.inject.Singleton
import android.util.Log

/**
 * Wrapper returned by [NovaCycleRepository.getCandles].
 *
 * @param candles            The candle bars for the requested (ticker, window, timeframe).
 * @param fromCache          True when the data was served from the Room cache because the
 *                           network fetch failed — i.e. the user is offline or the backend is
 *                           unreachable.  False for a live remote fetch.
 * @param newestBarTimestamp ISO-8601 timestamp of the most-recent bar in [candles], or null
 *                           if the list is empty.  Used by the UI to show "cached · last bar
 *                           X ago" when [fromCache] is true.
 */
data class CandlesWithSource(
    val candles: List<CandleResponse>,
    val fromCache: Boolean,
    val newestBarTimestamp: String?
)

/**
 * Single source of truth for all NovaCycle data.
 *
 * Strategy:
 *  1. Try the remote API first.
 *  2. On success, cache results in Room for offline access.
 *  3. On failure, fall back to the local cache.
 *  4. Return Result<T> so ViewModels can cleanly handle success/failure states.
 */
@Singleton
open class NovaCycleRepository @Inject constructor(
    private val apiService: NovaCycleApiService,
    private val signalDao: SignalDao,
    private val confidenceDao: ConfidenceDao,
    private val candleDao: CandleDao,
    private val freshnessTracker: DataFreshnessTracker
) {

    /**
     * Record a successful REMOTE fetch of user-visible data.
     * Deliberately NOT called for /healthz polls (reachability is not data
     * freshness) nor for Room cache fallbacks (cached data is not fresh).
     */
    private fun <T> T.recordDataFreshness(): T {
        freshnessTracker.recordSuccess()
        return this
    }

    suspend fun getPredictionLong(ticker: String = "VOO"): Result<PredictionResponse> =
        runCatching { apiService.predictLong(ticker).recordDataFreshness() }

    suspend fun getPredictionShort(ticker: String = "VOO"): Result<PredictionResponse> =
        runCatching { apiService.predictShort(ticker).recordDataFreshness() }

    suspend fun getHoldTime(ticker: String = "VOO"): Result<HoldTimeResponse> =
        runCatching { apiService.getHoldTime(ticker).recordDataFreshness() }

    suspend fun getConfidenceHistory(
        ticker: String = "VOO",
        window: String = "7d"
    ): Result<List<ConfidenceHistoryResponse>> = runCatching {
        val remote = apiService.getConfidenceHistory(ticker, window).recordDataFreshness()
        // Cache for offline access
        val entities = remote.map { r ->
            ConfidenceHistoryEntity(
                id = r.id,
                timestamp = r.timestamp,
                ticker = r.ticker,
                longBuyConfidence = r.longBuyConfidence,
                longSellConfidence = r.longSellConfidence,
                shortBuyConfidence = r.shortBuyConfidence,
                shortSellConfidence = r.shortSellConfidence,
                sessionType = r.sessionType,
                isExtendedHours = r.isExtendedHours
            )
        }
        confidenceDao.deleteByTicker(ticker)
        confidenceDao.insertAll(entities)
        remote
    }.recoverCatching { error ->
        // Fallback: serve from Room cache, re-map entities to response objects
        val cached = confidenceDao.getAllByTicker(ticker)
        if (cached.isEmpty()) throw error
        cached.map { e ->
            ConfidenceHistoryResponse(
                id = e.id,
                timestamp = e.timestamp,
                ticker = e.ticker,
                longBuyConfidence = e.longBuyConfidence,
                longSellConfidence = e.longSellConfidence,
                shortBuyConfidence = e.shortBuyConfidence,
                shortSellConfidence = e.shortSellConfidence,
                sessionType = e.sessionType,
                isExtendedHours = e.isExtendedHours
            )
        }
    }

    suspend fun getSignalHistory(
        ticker: String = "VOO",
        window: String = "30d"
    ): Result<List<SignalResponse>> = runCatching {
        val remote = apiService.getSignalHistory(ticker, window).recordDataFreshness()
        val entities = remote.map { r ->
            SignalHistoryEntity(
                id = r.id,
                timestamp = r.timestamp,
                ticker = r.ticker,
                cycleId = r.cycleId,
                signalType = r.signalType,
                gaugeType = r.gaugeType,
                confidence = r.confidence,
                sessionType = r.sessionType,
                isExtendedHours = r.isExtendedHours,
                gapType = r.gapType,
                liquidityScore = r.liquidityScore,
                macroOverrideApplied = r.macroOverrideApplied,
                convictionTier = r.convictionTier,
                // Reason sentences never contain newlines; store newline-joined.
                convictionReasons = r.convictionReasons
                    .takeIf { it.isNotEmpty() }?.joinToString("\n")
            )
        }
        signalDao.deleteByTicker(ticker)
        signalDao.insertAll(entities)
        remote
    }.recoverCatching { error ->
        val cached = signalDao.getAllByTicker(ticker)
        if (cached.isEmpty()) throw error
        cached.map { e ->
            SignalResponse(
                id = e.id,
                timestamp = e.timestamp,
                ticker = e.ticker,
                cycleId = e.cycleId,
                signalType = e.signalType,
                gaugeType = e.gaugeType,
                confidence = e.confidence,
                sessionType = e.sessionType,
                isExtendedHours = e.isExtendedHours,
                gapType = e.gapType,
                liquidityScore = e.liquidityScore,
                macroOverrideApplied = e.macroOverrideApplied,
                convictionTier = e.convictionTier,
                convictionReasons = e.convictionReasons
                    ?.split("\n")?.filter { it.isNotBlank() } ?: emptyList()
            )
        }
    }

    suspend fun getFilteredSignals(
        ticker: String = "VOO",
        window: String = "30d"
    ): Result<List<FilteredSignalResponse>> =
        runCatching { apiService.getFilteredSignalHistory(ticker, window).recordDataFreshness() }

    suspend fun getCandles(
        ticker: String = "VOO",
        window: String = "30d",
        timeframe: String = "daily"
    ): Result<CandlesWithSource> = runCatching {
        val remote = apiService.getVooCandles(ticker, window, timeframe).recordDataFreshness()
        // Cache is timeframe-aware: each (ticker, timeframe) series is stored
        // independently, so every timeframe survives offline without mixing
        // bar resolutions.
        val entities = remote.map { r ->
            CandleEntity(
                ticker = ticker,
                timeframe = timeframe,
                timestamp = r.timestamp,
                open = r.open,
                high = r.high,
                low = r.low,
                close = r.close,
                volume = r.volume,
                isExtendedHours = r.isExtendedHours,
                sessionType = r.sessionType,
                gapPercent = r.gapPercent,
                gapType = r.gapType
            )
        }
        candleDao.deleteByTickerAndTimeframe(ticker, timeframe)
        candleDao.insertAll(entities)
        CandlesWithSource(
            candles = remote,
            fromCache = false,
            newestBarTimestamp = remote.lastOrNull()?.timestamp
        )
    }.recoverCatching { error ->
        val cached = candleDao.getAllByTickerAndTimeframe(ticker, timeframe)
        if (cached.isEmpty()) throw error
        val candles = cached.map { e ->
            CandleResponse(
                timestamp = e.timestamp,
                open = e.open,
                high = e.high,
                low = e.low,
                close = e.close,
                volume = e.volume,
                isExtendedHours = e.isExtendedHours,
                sessionType = e.sessionType,
                gapPercent = e.gapPercent,
                gapType = e.gapType
            )
        }
        CandlesWithSource(
            candles = candles,
            fromCache = true,
            newestBarTimestamp = candles.lastOrNull()?.timestamp
        )
    }

    suspend fun getPriceSnapshot(
        ticker: String = "VOO"
    ): Result<PriceSnapshotResponse> =
        runCatching { apiService.getPriceSnapshot(ticker).recordDataFreshness() }

    suspend fun getIndicators(ticker: String = "VOO"): Result<IndicatorResponse> =
        runCatching { apiService.getIndicators(ticker).recordDataFreshness() }

    /** Current macro safety state (VIX regime + macro override status). */
    suspend fun getMacroSafety(ticker: String = "VOO"): Result<MacroSafetyResponse> =
        runCatching { apiService.getMacroSafety(ticker) }

    /** Backend health snapshot — used to surface the degraded-predictions warning. */
    open suspend fun getHealth(): Result<HealthzResponse> =
        runCatching { apiService.healthz() }

    /**
     * Fetch trade cycles and reliability metrics from /trade_history.
     * Kept as a simple remote call because the backend is the source of truth
     * for generated cycles and computed metrics.
     */
    open suspend fun getTradeHistory(
        ticker: String = "VOO",
        window: String = "30d"
    ): Result<TradeHistoryResponse> =
        runCatching { apiService.getTradeHistory(ticker, window).recordDataFreshness() }

    /**
     * Fetch model-performance analytics from /model_performance.
     * Kept as a simple remote call because the backend is the source of truth
     * for computed precision, calibration, streaks and missed-rally metrics.
     */
    open suspend fun getModelPerformance(
        ticker: String = "VOO",
        window: String = "90d",
        period: String = "day",
        confidenceMin: Float? = null,
        confidenceMax: Float? = null
    ): Result<ModelPerformanceResponse> =
        runCatching {
            apiService.getModelPerformance(ticker, window, period, confidenceMin, confidenceMax)
                .recordDataFreshness()
        }

    /**
     * Fetch realized per-conviction-tier performance from /tier_track_record.
     * Kept as a simple remote call because the backend is the source of truth
     * for cycle aggregation and sparse-sample handling.
     */
    open suspend fun getTierTrackRecord(
        ticker: String = "VOO",
        window: String = "90d"
    ): Result<TierTrackRecordResponse> =
        runCatching { apiService.getTierTrackRecord(ticker, window).recordDataFreshness() }

    /**
     * Check whether the backend still holds this token.
     *
     * Returns:
     *   Result.success(true)  — token is registered
     *   Result.success(false) — backend responded 404 (token missing, e.g. after DB reset)
     *   Result.failure(...)   — network/backend unreachable; caller should retry later
     */
    suspend fun checkDeviceToken(token: String): Result<Boolean> = runCatching {
        apiService.checkDeviceToken(token)
        true   // 2xx → token found
    }.recoverCatching { error ->
        // Retrofit throws HttpException for non-2xx responses
        val code = (error as? retrofit2.HttpException)?.code()
        if (code == 404) {
            Log.d("NovaCycleRepository", "Token not found on backend (DB may have been reset)")
            false
        } else {
            throw error   // propagate network errors so caller knows it couldn't reach backend
        }
    }

    /**
     * Register (or refresh) an FCM device token with the backend, including the
     * user's current notification preferences so the backend can filter signals
     * per-device before firing push notifications.
     *
     * Idempotent — the backend upserts by token value.
     *
     * @param token      FCM registration token
     * @param deviceName Human-readable device label (e.g. "Pixel 7")
     * @param settings   Current sensitivity settings; preferences default to lenient
     *                   values when null (all signals pass through)
     */
    suspend fun registerDeviceToken(
        token: String,
        deviceName: String? = null,
        settings: SensitivitySettings? = null
    ): Result<Unit> = runCatching {
        val (minBuy, minSell) = computeEffectiveThresholds(settings)
        val extHours = settings?.extendedHoursNotifications ?: true
        val highConvictionOnly = settings?.highConvictionOnly ?: false
        apiService.registerDeviceToken(
            RegisterDeviceRequest(
                token = token,
                deviceName = deviceName,
                minBuyThreshold = minBuy,
                minSellThreshold = minSell,
                extendedHoursNotifications = extHours,
                highConvictionOnly = highConvictionOnly,
            )
        )
        Log.d("NovaCycleRepository", "Device token registered: ${token.take(20)}... " +
            "(buyThreshold=${"%.2f".format(minBuy)}, sellThreshold=${"%.2f".format(minSell)}, " +
            "extHours=$extHours, highConvictionOnly=$highConvictionOnly)")
    }

    /**
     * Translate user-facing SensitivitySettings into backend-ready confidence thresholds.
     *
     * NotifSensitivity mapping:
     *   HIGH     → 0.50 (buzz for any signal ≥ 50 %)
     *   STANDARD → user's slider value (default 70 %)
     *   LOW      → 0.85 (only strong signals)
     *
     * @return Pair(minBuyThreshold, minSellThreshold) in [0.0, 1.0]
     */
    private fun computeEffectiveThresholds(settings: SensitivitySettings?): Pair<Double, Double> {
        if (settings == null) return Pair(0.70, 0.70)
        val threshold = when (settings.notificationSensitivity) {
            NotifSensitivity.HIGH     -> 0.50
            NotifSensitivity.LOW      -> 0.85
            NotifSensitivity.STANDARD -> settings.buyThreshold / 100.0
        }
        val sellThreshold = when (settings.notificationSensitivity) {
            NotifSensitivity.HIGH     -> 0.50
            NotifSensitivity.LOW      -> 0.85
            NotifSensitivity.STANDARD -> kotlin.math.abs(settings.sellThreshold) / 100.0
        }
        return Pair(threshold, sellThreshold)
    }
}
