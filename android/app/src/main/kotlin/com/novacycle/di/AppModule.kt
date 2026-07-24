package com.novacycle.di

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStore
import androidx.room.Room
import com.novacycle.data.local.NovaCycleDatabase
import com.novacycle.data.local.dao.CandleDao
import com.novacycle.data.local.dao.ConfidenceDao
import com.novacycle.data.local.dao.SignalDao
import com.novacycle.data.remote.NovaCycleApiService
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

    @Provides
    @Singleton
    fun provideNovaCycleDatabase(
        @ApplicationContext context: Context
    ): NovaCycleDatabase = Room.databaseBuilder(
        context,
        NovaCycleDatabase::class.java,
        "novacycle_db"
    )
        .fallbackToDestructiveMigration() // Simple strategy: rebuild on schema change
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
        candleDao: CandleDao
    ): NovaCycleRepository = NovaCycleRepository(apiService, signalDao, confidenceDao, candleDao)

    /** DataStore for user sensitivity settings — survives process death */
    @Provides
    @Singleton
    fun provideDataStore(
        @ApplicationContext context: Context
    ): DataStore<Preferences> = context.settingsDataStore
}
