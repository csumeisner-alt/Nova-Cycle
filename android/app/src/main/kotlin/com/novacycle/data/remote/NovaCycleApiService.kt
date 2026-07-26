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

    /** Retrieve historical trade records and reliability summary metrics */
    @GET("trade_history")
    suspend fun getTradeHistory(
        @Query("ticker") ticker: String = "VOO",
        @Query("window") window: String = "30d"
    ): TradeHistoryResponse

    /** Register or refresh an FCM device token with the backend */
    @POST("register_device")
    suspend fun registerDeviceToken(
        @Body request: RegisterDeviceRequest
    ): Map<String, Any>

    /** Remove a device token from the backend */
    @DELETE("unregister_device")
    suspend fun unregisterDeviceToken(
        @Query("token") token: String
    ): Map<String, Any>

    /** Send a test push notification to all registered devices */
    @POST("test_notification")
    suspend fun testNotification(
        @Body body: Map<String, Any> = emptyMap()
    ): Map<String, Any>

    /**
     * Check whether a token is known to the backend.
     * Returns 200 if found, throws HttpException(404) if not.
     * Used on every launch to detect a backend DB reset.
     */
    @GET("device_tokens/check")
    suspend fun checkDeviceToken(
        @Query("token") token: String
    ): Map<String, Any>

    /**
     * Server-side verification of a Google Play purchase token.
     * The backend calls the Play Developer API and records the entitlement.
     */
    @POST("billing/verify_purchase")
    suspend fun verifyPurchase(
        @Body request: VerifyPurchaseRequest
    ): EntitlementResponse

    /**
     * Re-check a previously verified purchase token.
     * Refunds issued since purchase are detected here (entitled=false, state=revoked).
     */
    @GET("billing/entitlement")
    suspend fun checkEntitlement(
        @Query("product_id") productId: String,
        @Query("purchase_token") purchaseToken: String
    ): EntitlementResponse

    /** Health check endpoint — reports "ok" or "degraded" plus per-model health */
    @GET("healthz")
    suspend fun healthz(): HealthzResponse
}
