package com.novacycle.data.local.dao

import androidx.room.*
import com.novacycle.data.local.entities.CandleEntity

@Dao
interface CandleDao {

    @Query("SELECT * FROM candles WHERE ticker = :ticker ORDER BY timestamp ASC")
    suspend fun getAllByTicker(ticker: String = "VOO"): List<CandleEntity>

    @Query("SELECT * FROM candles WHERE ticker = :ticker AND timestamp >= :since ORDER BY timestamp ASC")
    suspend fun getByTickerSince(ticker: String, since: String): List<CandleEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(candles: List<CandleEntity>)

    @Query("DELETE FROM candles WHERE ticker = :ticker")
    suspend fun deleteByTicker(ticker: String)

    @Query("DELETE FROM candles")
    suspend fun deleteAll()
}
