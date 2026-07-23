package com.novacycle.domain.model

/**
 * Domain model for a candlestick — UI-layer representation of OHLCV data.
 */
data class CandleData(
    val timestamp: String,
    val ticker: String = "VOO",
    val open: Float,
    val high: Float,
    val low: Float,
    val close: Float,
    val volume: Long = 0L,
    val isExtendedHours: Boolean = false,
    val sessionType: String = "regular",
    val gapPercent: Float? = null,
    val gapType: String? = null
) {
    /** Bullish candle: close above open */
    val isBullish: Boolean get() = close >= open
    val bodySize: Float get() = kotlin.math.abs(close - open)
    val range: Float get() = high - low
}
