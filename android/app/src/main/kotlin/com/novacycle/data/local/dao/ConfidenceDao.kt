package com.novacycle.data.local.dao

import androidx.room.*
import com.novacycle.data.local.entities.ConfidenceHistoryEntity

@Dao
interface ConfidenceDao {

    @Query("SELECT * FROM confidence_history WHERE ticker = :ticker ORDER BY timestamp ASC")
    suspend fun getAllByTicker(ticker: String = "VOO"): List<ConfidenceHistoryEntity>

    @Query("SELECT * FROM confidence_history WHERE ticker = :ticker AND timestamp >= :since ORDER BY timestamp ASC")
    suspend fun getByTickerSince(ticker: String, since: String): List<ConfidenceHistoryEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(entries: List<ConfidenceHistoryEntity>)

    @Query("DELETE FROM confidence_history WHERE ticker = :ticker")
    suspend fun deleteByTicker(ticker: String)

    @Query("DELETE FROM confidence_history")
    suspend fun deleteAll()
}
