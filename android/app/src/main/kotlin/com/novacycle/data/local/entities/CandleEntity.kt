package com.novacycle.data.local.entities

import androidx.room.ColumnInfo
import androidx.room.Entity

/**
 * Room entity caching OHLCV candle data locally.
 * Composite key is ticker + timeframe + timestamp so every timeframe
 * (daily, 1h, 15m, 5m, ...) caches independently and can be served offline
 * without mixing bar resolutions.
 */
@Entity(
    tableName = "candles",
    primaryKeys = ["ticker", "timeframe", "timestamp"]
)
data class CandleEntity(
    @ColumnInfo(name = "ticker") val ticker: String = "VOO",
    @ColumnInfo(name = "timeframe", defaultValue = "daily") val timeframe: String = "daily",
    @ColumnInfo(name = "timestamp") val timestamp: String,
    @ColumnInfo(name = "open") val open: Float,
    @ColumnInfo(name = "high") val high: Float,
    @ColumnInfo(name = "low") val low: Float,
    @ColumnInfo(name = "close") val close: Float,
    @ColumnInfo(name = "volume") val volume: Long = 0L,
    @ColumnInfo(name = "is_extended_hours") val isExtendedHours: Boolean = false,
    @ColumnInfo(name = "session_type") val sessionType: String = "regular",
    @ColumnInfo(name = "gap_percent") val gapPercent: Float? = null,
    @ColumnInfo(name = "gap_type") val gapType: String? = null
)
