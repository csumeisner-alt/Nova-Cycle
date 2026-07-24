package com.novacycle.data.local.dao

import androidx.room.*
import com.novacycle.data.local.entities.SignalHistoryEntity

@Dao
interface SignalDao {

    @Query("SELECT * FROM signal_history WHERE ticker = :ticker ORDER BY timestamp DESC")
    suspend fun getAllByTicker(ticker: String = "VOO"): List<SignalHistoryEntity>

    @Query("SELECT * FROM signal_history WHERE ticker = :ticker AND timestamp >= :since ORDER BY timestamp ASC")
    suspend fun getByTickerSince(ticker: String, since: String): List<SignalHistoryEntity>

    /** Insert or replace on conflict — handles re-fetch of same signals */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(signals: List<SignalHistoryEntity>)

    @Query("DELETE FROM signal_history WHERE ticker = :ticker")
    suspend fun deleteByTicker(ticker: String)

    @Query("DELETE FROM signal_history")
    suspend fun deleteAll()
}
