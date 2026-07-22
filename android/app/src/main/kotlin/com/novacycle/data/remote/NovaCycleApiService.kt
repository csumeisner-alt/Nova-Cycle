package com.novacycle.data.remote

import com.novacycle.data.remote.models.*
import retrofit2.http.*

/**
 * Retrofit interface for all NovaCycle backend endpoints.
 * Base URL is configured in NetworkModule from BuildConfig.API_BASE_URL.
 * All functions are suspend — called from coroutines in ViewModels/Repository.
 */
interface NovaCycleApiService {

    /** Trigger long-trend prediction calculation and return result */
    @POST("predict_long")
    suspend fun predictLong(
        @Query("ticker") ticker: String = "VOO"
    ): PredictionResponse

    /** Trigger short-trend prediction calculation and return result */
    @POST("predict_short")
    suspend fun predictShort(
        @Query("ticker") ticker: String = "VOO"
    ): PredictionResponse

    /** Estimate how long to hold the current position */
    @POST("hold_time_estimate")
    suspend fun getHoldTime(
        @Query("ticker") ticker: String = "VOO"
    ): HoldTimeResponse

    /** Retrieve confidence history over a rolling time window */
    @GET("confidence_history")
    suspend fun getConfidenceHistory(
        @Query("ticker") ticker: String = "VOO",
        @Query("window") window: String = "7d"
    ): List<ConfidenceHistoryResponse>

    /** Retrieve all raw signal events over a rolling time window */
    @GET("signal_history")
    suspend fun getSignalHistory(
        @Query("ticker") ticker: String = "VOO",
        @Query("window") window: String = "30d"
    ): List<SignalResponse>

    /** Retrieve backend-filtered signals (strongest-confidence rule applied server-side) */
    @GET("filtered_signal_history")
    suspend fun getFilteredSignalHistory(
        @Query("ticker") ticker: String = "VOO",
        @Query("window") window: String = "30d"
    ): List<FilteredSignalResponse>

    /** Retrieve OHLCV candlestick data */
    @GET("voo_candles")
    suspend fun getVooCandles(
        @Query("ticker") ticker: String = "VOO",
        @Query("window") window: String = "30d",
        @Query("timeframe") timeframe: String = "daily"
    ): List<CandleResponse>

    /** Retrieve current technical indicator snapshot */
    @GET("indicators")
    suspend fun getIndicators(
        @Query("ticker") ticker: String = "VOO"
    ): IndicatorResponse

    /** Retrieve gap status for today's session */
    @GET("gap_status")
    suspend fun getGapStatus(
        @Query("ticker") ticker: String = "VOO"
    ): Map<String, Any>

    /** Retrieve historical trade records for reliability metrics */
    @GET("trade_history")
    suspend fun getTradeHistory(
        @Query("ticker") ticker: String = "VOO"
    ): List<Map<String, Any>>

    /** Health check endpoint */
    @GET("healthz")
    suspend fun healthz(): Map<String, Any>
}
