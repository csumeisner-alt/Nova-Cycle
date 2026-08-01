package com.novacycle.data.repository

import com.novacycle.data.local.dao.CandleDao
import com.novacycle.data.local.dao.ConfidenceDao
import com.novacycle.data.local.dao.SignalDao
import com.novacycle.data.local.entities.CandleEntity
import com.novacycle.data.remote.NovaCycleApiService
import com.novacycle.data.remote.models.CandleResponse
import io.mockk.coEvery
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.IOException

/**
 * Verifies that the candle cache is timeframe-aware:
 *  - every timeframe (daily + intraday) is cached after a successful fetch
 *  - every timeframe is served from cache when the network fails
 *  - timeframes never mix in the fallback
 */
class NovaCycleRepositoryCandleCacheTest {

    /** In-memory CandleDao mirroring the Room queries' (ticker, timeframe) scoping. */
    private class FakeCandleDao : CandleDao {
        val rows = mutableListOf<CandleEntity>()

        override suspend fun getAllByTickerAndTimeframe(
            ticker: String,
            timeframe: String
        ): List<CandleEntity> =
            rows.filter { it.ticker == ticker && it.timeframe == timeframe }
                .sortedBy { it.timestamp }

        override suspend fun getByTickerAndTimeframeSince(
            ticker: String,
            timeframe: String,
            since: String
        ): List<CandleEntity> =
            getAllByTickerAndTimeframe(ticker, timeframe).filter { it.timestamp >= since }

        override suspend fun insertAll(candles: List<CandleEntity>) {
            candles.forEach { c ->
                rows.removeAll {
                    it.ticker == c.ticker && it.timeframe == c.timeframe && it.timestamp == c.timestamp
                }
                rows.add(c)
            }
        }

        override suspend fun deleteByTickerAndTimeframe(ticker: String, timeframe: String) {
            rows.removeAll { it.ticker == ticker && it.timeframe == timeframe }
        }

        override suspend fun deleteAll() = rows.clear()
    }

    private lateinit var apiService: NovaCycleApiService
    private lateinit var candleDao: FakeCandleDao
    private lateinit var repository: NovaCycleRepository

    private fun candle(ts: String, close: Float) = CandleResponse(
        timestamp = ts, open = close - 1f, high = close + 1f, low = close - 2f, close = close
    )

    @Before
    fun setUp() {
        apiService = mockk()
        candleDao = FakeCandleDao()
        repository = NovaCycleRepository(
            apiService = apiService,
            signalDao = mockk(relaxed = true),
            confidenceDao = mockk(relaxed = true),
            candleDao = candleDao,
            freshnessTracker = DataFreshnessTracker()
        )
    }

    @Test
    fun `successful fetch caches every timeframe independently`() = runTest {
        coEvery { apiService.getVooCandles("VOO", "1d", "5m") } returns
            listOf(candle("2026-08-01T09:30:00", 100f))
        coEvery { apiService.getVooCandles("VOO", "30d", "daily") } returns
            listOf(candle("2026-07-31", 200f), candle("2026-08-01", 201f))

        val intradayResult = repository.getCandles("VOO", "1d", "5m")
        val dailyResult    = repository.getCandles("VOO", "30d", "daily")

        // Remote fetches must not be marked as coming from cache
        assertTrue("live intraday fetch should have fromCache=false", !intradayResult.getOrThrow().fromCache)
        assertTrue("live daily fetch should have fromCache=false",    !dailyResult.getOrThrow().fromCache)
        assertEquals("2026-08-01T09:30:00", intradayResult.getOrThrow().newestBarTimestamp)
        assertEquals("2026-08-01",          dailyResult.getOrThrow().newestBarTimestamp)

        assertEquals(1, candleDao.getAllByTickerAndTimeframe("VOO", "5m").size)
        assertEquals(2, candleDao.getAllByTickerAndTimeframe("VOO", "daily").size)
        assertEquals("5m", candleDao.getAllByTickerAndTimeframe("VOO", "5m").single().timeframe)
    }

    @Test
    fun `offline fallback serves cached intraday timeframes`() = runTest {
        for (tf in listOf("5m", "15m", "1h", "daily")) {
            coEvery { apiService.getVooCandles("VOO", "1d", tf) } returns
                listOf(candle("2026-08-01T09:30:00", 100f))
            repository.getCandles("VOO", "1d", tf)
        }
        // Go offline
        coEvery { apiService.getVooCandles(any(), any(), any()) } throws IOException("offline")

        for (tf in listOf("5m", "15m", "1h", "daily")) {
            val result = repository.getCandles("VOO", "1d", tf)
            assertTrue("expected cached fallback for $tf", result.isSuccess)
            assertEquals(1, result.getOrThrow().candles.size)
            assertTrue("expected fromCache=true for $tf", result.getOrThrow().fromCache)
        }
    }

    @Test
    fun `offline fallback never mixes timeframes`() = runTest {
        coEvery { apiService.getVooCandles("VOO", "30d", "daily") } returns
            listOf(candle("2026-07-31", 200f))
        repository.getCandles("VOO", "30d", "daily")

        coEvery { apiService.getVooCandles(any(), any(), any()) } throws IOException("offline")

        // 5m was never cached — must fail, not serve daily bars
        assertTrue(repository.getCandles("VOO", "1d", "5m").isFailure)
        // daily still succeeds from cache
        assertTrue(repository.getCandles("VOO", "30d", "daily").isSuccess)
    }

    @Test
    fun `refetching a timeframe replaces only that timeframe`() = runTest {
        coEvery { apiService.getVooCandles("VOO", "1d", "5m") } returns
            listOf(candle("2026-08-01T09:30:00", 100f), candle("2026-08-01T09:35:00", 101f))
        coEvery { apiService.getVooCandles("VOO", "30d", "daily") } returns
            listOf(candle("2026-08-01", 200f))
        repository.getCandles("VOO", "1d", "5m")
        repository.getCandles("VOO", "30d", "daily")

        // New 5m fetch returns a fresh, shorter series
        coEvery { apiService.getVooCandles("VOO", "1d", "5m") } returns
            listOf(candle("2026-08-01T10:00:00", 102f))
        repository.getCandles("VOO", "1d", "5m")

        assertEquals(1, candleDao.getAllByTickerAndTimeframe("VOO", "5m").size)
        assertEquals(1, candleDao.getAllByTickerAndTimeframe("VOO", "daily").size)
    }
}
