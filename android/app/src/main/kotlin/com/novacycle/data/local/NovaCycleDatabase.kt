package com.novacycle.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import com.novacycle.data.local.dao.CandleDao
import com.novacycle.data.local.dao.ConfidenceDao
import com.novacycle.data.local.dao.SignalDao
import com.novacycle.data.local.entities.CandleEntity
import com.novacycle.data.local.entities.ConfidenceHistoryEntity
import com.novacycle.data.local.entities.SignalHistoryEntity

/**
 * Room database definition.
 * Version is incremented on schema changes. v1→v2 adds a timeframe column to
 * candles (see MIGRATION_1_2 in AppModule); v2→v3 adds conviction-tier columns
 * to signal_history (MIGRATION_2_3); fallbackToDestructiveMigration
 * remains as a last-resort safety net.
 */
@Database(
    entities = [
        SignalHistoryEntity::class,
        ConfidenceHistoryEntity::class,
        CandleEntity::class
    ],
    version = 3,
    exportSchema = true
)
abstract class NovaCycleDatabase : RoomDatabase() {
    abstract fun signalDao(): SignalDao
    abstract fun confidenceDao(): ConfidenceDao
    abstract fun candleDao(): CandleDao
}
