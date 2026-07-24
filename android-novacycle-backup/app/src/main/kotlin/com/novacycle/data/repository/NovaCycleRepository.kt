package com.novacycle.data.repository

import com.novacycle.data.local.dao.CandleDao
import com.novacycle.data.local.dao.ConfidenceDao
import com.novacycle.data.local.dao.SignalDao
import com.novacycle.data.local.entities.CandleEntity
import com.novacycle.data.local.entities.ConfidenceHistoryEntity
import com.novacycle.data.local.entities.SignalHistoryEntity
import com.novacycle.data.remote.NovaCycleApiService
import com.novacycle.data.remote.models.*
import javax.inject.Inject
import javax.inject.Singleton

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
class NovaCycleRepository @Inject constructor(
    private val apiService: NovaCycleApiService,
    private val signalDao: SignalDao,
    private val confidenceDao: ConfidenceDao,
    private val candleDao: CandleDao
) {

    suspend fun getPredictionLong(ticker: String = "VOO"): Result<PredictionResponse> =
        runCatching { apiService.predictLong(ticker) }

    suspend fun getPredictionShort(ticker: String = "VOO"): Result<PredictionResponse> =
        runCatching { apiService.predictShort(ticker) }

    suspend fun getHoldTime(ticker: String = "VOO"): Result<HoldTimeResponse> =
        runCatching { apiService.getHoldTime(ticker) }

    suspend fun getConfidenceHistory(
        ticker: String = "VOO",
        window: String = "7d"
    ): Result<List<ConfidenceHistoryResponse>> = runCatching {
        val remote = apiService.getConfidenceHistory(ticker, window)
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
        val remote = apiService.getSignalHistory(ticker, window)
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
                macroOverrideApplied = r.macroOverrideApplied
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
                macroOverrideApplied = e.macroOverrideApplied
            )
        }
    }

    suspend fun getFilteredSignals(
        ticker: String = "VOO",
        window: String = "30d"
    ): Result<List<FilteredSignalResponse>> =
        runCatching { apiService.getFilteredSignalHistory(ticker, window) }

    suspend fun getCandles(
        ticker: String = "VOO",
        window: String = "30d"
    ): Result<List<CandleResponse>> = runCatching {
        val remote = apiService.getVooCandles(ticker, window)
        val entities = remote.map { r ->
            CandleEntity(
                ticker = ticker,
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
        candleDao.deleteByTicker(ticker)
        candleDao.insertAll(entities)
        remote
    }.recoverCatching { error ->
        val cached = candleDao.getAllByTicker(ticker)
        if (cached.isEmpty()) throw error
        cached.map { e ->
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
    }

    suspend fun getIndicators(ticker: String = "VOO"): Result<IndicatorResponse> =
        runCatching { apiService.getIndicators(ticker) }

    /**
     * Fetch trade cycles and reliability metrics from /trade_history.
     * Kept as a simple remote call because the backend is the source of truth
     * for generated cycles and computed metrics.
     */
    suspend fun getTradeHistory(
        ticker: String = "VOO",
        window: String = "30d"
    ): Result<TradeHistoryResponse> =
        runCatching { apiService.getTradeHistory(ticker, window) }
}
