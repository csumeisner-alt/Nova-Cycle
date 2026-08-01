package com.novacycle.viewmodel

import android.content.Context
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.NetworkMonitor
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.data.remote.models.FilteredSignalResponse
import com.novacycle.data.repository.CandlesWithSource
import com.novacycle.data.repository.ChartPreferencesRepository
import com.novacycle.data.repository.ChartScreenKey
import com.novacycle.data.repository.DataFreshnessTracker
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.usecase.ApplyFilteredSignalsUseCase
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Tests that [FilteredChartViewModel] auto-refreshes when the network comes back
 * while the offline-cache badge is showing:
 *
 *  - Network reconnect while badge is visible → background loadData() fires
 *  - Successful fetch after reconnect → candlesFromCache cleared (badge gone)
 *  - Network reconnect while badge is NOT visible → no extra reload
 *  - Rapid reconnects within the debounce window → only one reload
 *  - Sustained offline state → no spurious reload
 */
@OptIn(ExperimentalCoroutinesApi::class)
class FilteredChartViewModelNetworkReconnectTest {

    private val dispatcher = StandardTestDispatcher()

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    // ── Fakes ─────────────────────────────────────────────────────────────────

    private class FakeNetworkMonitor : NetworkMonitor(mockk<Context>()) {
        val events = MutableSharedFlow<Boolean>(replay = 1)
        override val isConnected: Flow<Boolean> = events
    }

    private class FakeRepository(
        private val candleResults: List<Result<CandlesWithSource>> = listOf(
            Result.success(CandlesWithSource(emptyList(), false, null))
        )
    ) : NovaCycleRepository(
        apiService = mockk(),
        signalDao = mockk(relaxed = true),
        confidenceDao = mockk(relaxed = true),
        candleDao = mockk(relaxed = true),
        freshnessTracker = DataFreshnessTracker()
    ) {
        var candleCallCount = 0

        override suspend fun getCandles(
            ticker: String,
            window: String,
            timeframe: String
        ): Result<CandlesWithSource> {
            val r = candleResults[minOf(candleCallCount, candleResults.size - 1)]
            candleCallCount++
            return r
        }

        override suspend fun getFilteredSignals(
            ticker: String,
            window: String
        ): Result<List<FilteredSignalResponse>> = Result.success(emptyList())

        override suspend fun getPriceSnapshot(
            ticker: String
        ) = Result.failure<com.novacycle.data.remote.models.PriceSnapshotResponse>(
            RuntimeException("not needed")
        )
    }

    private class FakeChartPrefs : ChartPreferencesRepository(
        dataStore = mockk(relaxed = true)
    ) {
        override fun prefs(screen: ChartScreenKey) = flowOf(
            com.novacycle.data.repository.ChartPrefs()
        )
        override suspend fun saveTimeframe(screen: ChartScreenKey, timeframe: String) = Unit
        override suspend fun saveRenderMode(screen: ChartScreenKey, renderMode: String) = Unit
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun _seedCacheState(vm: FilteredChartViewModel) {
        val field = vm.javaClass.getDeclaredField("_uiState")
        field.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        val flow = field.get(vm)
            as kotlinx.coroutines.flow.MutableStateFlow<FilteredChartUiState>
        flow.value = flow.value.copy(
            candles = listOf(
                CandleResponse(
                    timestamp = "2024-01-01",
                    open = 400f, high = 405f, low = 395f, close = 402f
                )
            ),
            candlesFromCache = true,
            cacheNewestBarTimestamp = "2024-01-01"
        )
    }

    private fun buildViewModel(
        repository: FakeRepository = FakeRepository(),
        monitor: FakeNetworkMonitor = FakeNetworkMonitor()
    ): FilteredChartViewModel =
        FilteredChartViewModel(repository, ApplyFilteredSignalsUseCase(), FakeChartPrefs(), monitor)

    private fun vmTest(
        repository: FakeRepository = FakeRepository(),
        monitor: FakeNetworkMonitor = FakeNetworkMonitor(),
        block: suspend TestScope.(FilteredChartViewModel, FakeNetworkMonitor, FakeRepository) -> Unit
    ) = runTest(dispatcher) {
        val vm = buildViewModel(repository, monitor)
        try {
            block(vm, monitor, repository)
        } finally {
            vm.viewModelScope.cancel()
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    @Test
    fun `reconnect while badge is visible triggers a background loadData`() = vmTest { vm, monitor, repo ->
        advanceUntilIdle()

        _seedCacheState(vm)
        assertTrue("precondition: badge must be showing", vm.uiState.value.candlesFromCache)

        val callsBefore = repo.candleCallCount

        monitor.events.emit(true)
        advanceTimeBy(FilteredChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
        advanceUntilIdle()

        assertTrue(
            "loadData should have fired at least once after reconnect",
            repo.candleCallCount > callsBefore
        )
    }

    @Test
    fun `successful fetch after reconnect clears the cache badge`() = vmTest(
        repository = FakeRepository(
            candleResults = listOf(
                // init call → cached data
                Result.success(CandlesWithSource(
                    candles = listOf(CandleResponse(
                        timestamp = "2024-01-01",
                        open = 400f, high = 405f, low = 395f, close = 402f
                    )),
                    fromCache = true,
                    newestBarTimestamp = "2024-01-01"
                )),
                // reconnect call → live data
                Result.success(CandlesWithSource(
                    candles = listOf(CandleResponse(
                        timestamp = "2024-01-02",
                        open = 403f, high = 408f, low = 400f, close = 406f
                    )),
                    fromCache = false,
                    newestBarTimestamp = "2024-01-02"
                ))
            )
        )
    ) { vm, monitor, _ ->
        advanceUntilIdle()

        assertTrue("precondition: badge visible after cached load", vm.uiState.value.candlesFromCache)

        monitor.events.emit(true)
        advanceTimeBy(FilteredChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
        advanceUntilIdle()

        assertFalse("badge must be gone after live fetch", vm.uiState.value.candlesFromCache)
        assertEquals(null, vm.uiState.value.cacheNewestBarTimestamp)
    }

    @Test
    fun `reconnect while badge is NOT visible does not trigger extra load`() = vmTest(
        repository = FakeRepository(
            candleResults = listOf(
                Result.success(CandlesWithSource(emptyList(), fromCache = false, null))
            )
        )
    ) { vm, monitor, repo ->
        advanceUntilIdle()

        assertFalse("precondition: no cache badge", vm.uiState.value.candlesFromCache)
        val callsBefore = repo.candleCallCount

        monitor.events.emit(true)
        advanceTimeBy(FilteredChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
        advanceUntilIdle()

        assertEquals(
            "no extra loadData should fire when badge is not showing",
            callsBefore,
            repo.candleCallCount
        )
    }

    @Test
    fun `rapid reconnects within debounce window trigger only one reload`() = vmTest { vm, monitor, repo ->
        advanceUntilIdle()
        _seedCacheState(vm)

        val callsBefore = repo.candleCallCount

        monitor.events.emit(true)
        advanceTimeBy(100)
        monitor.events.emit(false)
        advanceTimeBy(100)
        monitor.events.emit(true)
        advanceTimeBy(100)
        monitor.events.emit(false)
        advanceTimeBy(100)
        monitor.events.emit(true)

        advanceTimeBy(FilteredChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
        advanceUntilIdle()

        assertEquals(
            "debounce should collapse rapid reconnects into exactly one reload",
            callsBefore + 1,
            repo.candleCallCount
        )
    }

    @Test
    fun `sustained offline state emits no reconnect and triggers no reload`() = vmTest { vm, monitor, repo ->
        advanceUntilIdle()
        _seedCacheState(vm)

        val callsBefore = repo.candleCallCount

        monitor.events.emit(false)
        advanceTimeBy(FilteredChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
        advanceUntilIdle()

        assertEquals(
            "offline state should not trigger any loadData",
            callsBefore,
            repo.candleCallCount
        )
    }
}
