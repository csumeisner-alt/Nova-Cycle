package com.novacycle.di

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStore
import androidx.room.Room
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.novacycle.data.local.NovaCycleDatabase
import com.novacycle.data.local.dao.CandleDao
import com.novacycle.data.local.dao.ConfidenceDao
import com.novacycle.data.local.dao.SignalDao
import com.novacycle.data.remote.NovaCycleApiService
import com.novacycle.data.repository.DataFreshnessTracker
import com.novacycle.data.repository.NovaCycleRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

// DataStore extension property — one DataStore instance per app
private val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(
    name = "novacycle_settings"
)

/**
 * Hilt module providing Room database, DAOs, Repository, and DataStore.
 * All are singletons to avoid database connection churn.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    /**
     * v1 → v2: candles gains a `timeframe` column and its primary key becomes
     * (ticker, timeframe, timestamp). SQLite cannot alter primary keys, so the
     * table is rebuilt; existing rows were all daily bars and are preserved as
     * timeframe='daily'.
     */
    val MIGRATION_1_2 = object : Migration(1, 2) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS candles_new (
                    ticker TEXT NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT 'daily',
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    is_extended_hours INTEGER NOT NULL,
                    session_type TEXT NOT NULL,
                    gap_percent REAL,
                    gap_type TEXT,
                    PRIMARY KEY(ticker, timeframe, timestamp)
                )
                """.trimIndent()
            )
            db.execSQL(
                """
                INSERT INTO candles_new (
                    ticker, timeframe, timestamp, open, high, low, close,
                    volume, is_extended_hours, session_type, gap_percent, gap_type
                )
                SELECT ticker, 'daily', timestamp, open, high, low, close,
                       volume, is_extended_hours, session_type, gap_percent, gap_type
                FROM candles
                """.trimIndent()
            )
            db.execSQL("DROP TABLE candles")
            db.execSQL("ALTER TABLE candles_new RENAME TO candles")
        }
    }

    /**
     * v2 → v3: signal_history gains nullable conviction-tier columns.
     */
    val MIGRATION_2_3 = object : Migration(2, 3) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("ALTER TABLE signal_history ADD COLUMN conviction_tier TEXT")
            db.execSQL("ALTER TABLE signal_history ADD COLUMN conviction_reasons TEXT")
        }
    }

    @Provides
    @Singleton
    fun provideNovaCycleDatabase(
        @ApplicationContext context: Context
    ): NovaCycleDatabase = Room.databaseBuilder(
        context,
        NovaCycleDatabase::class.java,
        "novacycle_db"
    )
        .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
        .fallbackToDestructiveMigration() // Last-resort safety net for unknown versions
        .build()

    @Provides
    @Singleton
    fun provideSignalDao(db: NovaCycleDatabase): SignalDao = db.signalDao()

    @Provides
    @Singleton
    fun provideConfidenceDao(db: NovaCycleDatabase): ConfidenceDao = db.confidenceDao()

    @Provides
    @Singleton
    fun provideCandleDao(db: NovaCycleDatabase): CandleDao = db.candleDao()

    @Provides
    @Singleton
    fun provideNovaCycleRepository(
        apiService: NovaCycleApiService,
        signalDao: SignalDao,
        confidenceDao: ConfidenceDao,
        candleDao: CandleDao,
        freshnessTracker: DataFreshnessTracker
    ): NovaCycleRepository =
        NovaCycleRepository(apiService, signalDao, confidenceDao, candleDao, freshnessTracker)

    /** DataStore for user sensitivity settings — survives process death */
    @Provides
    @Singleton
    fun provideDataStore(
        @ApplicationContext context: Context
    ): DataStore<Preferences> = context.settingsDataStore
}
