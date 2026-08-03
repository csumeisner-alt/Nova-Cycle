package com.novacycle.data.local.entities

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity caching signal history locally.
 * Used as an offline fallback when the API is unreachable.
 */
@Entity(tableName = "signal_history")
data class SignalHistoryEntity(
    @PrimaryKey
    @ColumnInfo(name = "id") val id: String,
    @ColumnInfo(name = "timestamp") val timestamp: String,
    @ColumnInfo(name = "ticker") val ticker: String = "VOO",
    @ColumnInfo(name = "cycle_id") val cycleId: String? = null,
    @ColumnInfo(name = "signal_type") val signalType: String,
    @ColumnInfo(name = "gauge_type") val gaugeType: String,
    @ColumnInfo(name = "confidence") val confidence: Float,
    @ColumnInfo(name = "session_type") val sessionType: String = "regular",
    @ColumnInfo(name = "is_extended_hours") val isExtendedHours: Boolean = false,
    @ColumnInfo(name = "gap_type") val gapType: String? = null,
    @ColumnInfo(name = "liquidity_score") val liquidityScore: Float = 1f,
    @ColumnInfo(name = "macro_override_applied") val macroOverrideApplied: Boolean = false,
    /** "opportunity" | "high_conviction" | null for pre-tiering rows. */
    @ColumnInfo(name = "conviction_tier") val convictionTier: String? = null,
    /** JSON-encoded list of reason strings (kept opaque in the cache). */
    @ColumnInfo(name = "conviction_reasons") val convictionReasons: String? = null,
    /** Model reliability when stored; null for pre-column rows (unknown). */
    @ColumnInfo(name = "model_state") val modelState: String? = null
)
