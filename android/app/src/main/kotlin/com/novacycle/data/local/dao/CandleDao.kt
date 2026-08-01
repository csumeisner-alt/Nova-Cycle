package com.novacycle.data.local.dao

import androidx.room.*
import com.novacycle.data.local.entities.CandleEntity

@Dao
interface CandleDao {

    @Query("SELECT * FROM candles WHERE ticker = :ticker AND timeframe = :timeframe ORDER BY timestamp ASC")
    suspend fun getAllByTickerAndTimeframe(ticker: String = "VOO", timeframe: String = "daily"): List<CandleEntity>

    @Query("SELECT * FROM candles WHERE ticker = :ticker AND timeframe = :timeframe AND timestamp >= :since ORDER BY timestamp ASC")
    suspend fun getByTickerAndTimeframeSince(ticker: String, timeframe: String, since: String): List<CandleEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(candles: List<CandleEntity>)

    @Query("DELETE FROM candles WHERE ticker = :ticker AND timeframe = :timeframe")
    suspend fun deleteByTickerAndTimeframe(ticker: String, timeframe: String)

    @Query("DELETE FROM candles")
    suspend fun deleteAll()
}
