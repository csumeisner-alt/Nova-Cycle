package com.novacycle.data.local.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity caching confidence history snapshots locally.
 */
@Entity(tableName = "confidence_history")
data class ConfidenceHistoryEntity(
    @PrimaryKey
    @ColumnInfo(name = "id") val id: String,
    @ColumnInfo(name = "timestamp") val timestamp: String,
    @ColumnInfo(name = "ticker") val ticker: String = "VOO",
    @ColumnInfo(name = "long_buy_confidence") val longBuyConfidence: Float,
    @ColumnInfo(name = "long_sell_confidence") val longSellConfidence: Float,
    @ColumnInfo(name = "short_buy_confidence") val shortBuyConfidence: Float,
    @ColumnInfo(name = "short_sell_confidence") val shortSellConfidence: Float,
    @ColumnInfo(name = "session_type") val sessionType: String = "regular",
    @ColumnInfo(name = "is_extended_hours") val isExtendedHours: Boolean = false
)
