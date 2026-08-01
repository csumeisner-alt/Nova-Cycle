package com.novacycle.viewmodel

import android.content.Context
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.NetworkMonitor
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.data.repository.CandlesWithSource
import com.novacycle.data.repository.ChartPreferencesRepository
import com.novacycle.data.repository.ChartScreenKey
import com.novacycle.data.repository.DataFreshnessTracker
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.usecase.GetSignalsUseCase
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
 * Tests that [RawChartViewModel] auto-refreshes when the network comes back
 * while the offline-cache badge is showing:
 *
 *  - Network reconnect while badge is visible → background loadData() fires
 *  - Successful fetch after reconnect → candlesFromCache cleared (badge gone)
 *  - Network reconnect while badge is NOT visible → no extra reload
 *  - Rapid reconnects within the debounce window → only one reload
 *  - Sustained offline state → no spurious reload
 */
@OptIn(ExperimentalCoroutinesApi::class)
class RawChartViewModelNetworkReconnectTest {

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

    /**
     * Fake [NetworkMonitor] that exposes a [MutableSharedFlow] to drive
     * connectivity events in tests.  Subclasses the real class to satisfy the
     * constructor; [isConnected] is overridden so the mock context is never used.
     */
    private class FakeNetworkMonitor : NetworkMonitor(mockk<Context>()) {
        val events = MutableSharedFlow<Boolean>(replay = 1)
        override val isConnected: Flow<Boolean> = events
    }

    /**
     * Configurable fake repository.  [candleResults] are consumed in order;
     * the last one is repeated once exhausted.
     */
    private class FakeRepository(
        private val candleResults: List<Result<CandlesWithSource>> = listOf(
            Result.success(CandlesWithSource(emptyList(), false, null))
        ),
        private val signalResults: List<Result<List<com.novacycle.data.remote.models.SignalResponse>>> = listOf(
            Result.success(emptyList())
        )
    ) : NovaCycleRepository(
        apiService = mockk(),
        signalDao = mockk(relaxed = true),
        confidenceDao = mockk(relaxed = true),
        candleDao = mockk(relaxed = true),
        freshnessTracker = DataFreshnessTracker()
    ) {
        var candleCallCount = 0
        private var sigCallCount = 0

        override suspend fun getCandles(
            ticker: String,
            window: String,
            timeframe: String
        ): Result<CandlesWithSource> {
            val r = candleResults[minOf(candleCallCount, candleResults.size - 1)]
            candleCallCount++
            return r
        }

        override suspend fun getSignalHistory(
            ticker: String,
            window: String
        ): Result<List<com.novacycle.data.remote.models.SignalResponse>> {
            val r = signalResults[minOf(sigCallCount, signalResults.size - 1)]
            sigCallCount++
            return r
        }

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

    /** Pre-seed ViewModel state as if a previous load returned cached data. */
    private fun RawChartViewModel.seedCacheState() {
        uiState  // access to confirm it's initialised
        _seedCacheState(this)
    }

    private fun _seedCacheState(vm: RawChartViewModel) {
        // Force the UI state to look like an offline cache load
        val field = vm.javaClass.getDeclaredField("_uiState")
        field.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        val flow = field.get(vm) as kotlinx.coroutines.flow.MutableStateFlow<RawChartUiState>
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
    ): RawChartViewModel {
        val useCase = GetSignalsUseCase(repository)
        return RawChartViewModel(repository, useCase, FakeChartPrefs(), monitor)
    }

    private fun vmTest(
        repository: FakeRepository = FakeRepository(),
        monitor: FakeNetworkMonitor = FakeNetworkMonitor(),
        block: suspend TestScope.(RawChartViewModel, FakeNetworkMonitor, FakeRepository) -> Unit
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
        advanceUntilIdle() // let init load finish

        // Manually put the VM in offline-cache state
        _seedCacheState(vm)
        assertTrue("precondition: badge must be showing", vm.uiState.value.candlesFromCache)

        val callsBefore = repo.candleCallCount

        // Emit reconnect, then advance past the debounce window
        monitor.events.emit(true)
        advanceTimeBy(RawChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
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
                // First (init) call returns cached data
                Result.success(CandlesWithSource(
                    candles = listOf(CandleResponse(
                        timestamp = "2024-01-01",
                        open = 400f, high = 405f, low = 395f, close = 402f
                    )),
                    fromCache = true,
                    newestBarTimestamp = "2024-01-01"
                )),
                // Second (reconnect) call returns live data
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
        advanceUntilIdle() // init load → candlesFromCache = true

        assertTrue("precondition: badge visible after cached load", vm.uiState.value.candlesFromCache)

        monitor.events.emit(true)
        advanceTimeBy(RawChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
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
        advanceTimeBy(RawChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
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

        // Emit several events quickly — all within the 500 ms debounce
        monitor.events.emit(true)
        advanceTimeBy(100)
        monitor.events.emit(false)
        advanceTimeBy(100)
        monitor.events.emit(true)
        advanceTimeBy(100)
        monitor.events.emit(false)
        advanceTimeBy(100)
        monitor.events.emit(true)

        // Now let the debounce window expire
        advanceTimeBy(RawChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
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

        // Keep emitting offline — never reconnect
        monitor.events.emit(false)
        advanceTimeBy(RawChartViewModel.RECONNECT_DEBOUNCE_MS + 100)
        advanceUntilIdle()

        assertEquals(
            "offline state should not trigger any loadData",
            callsBefore,
            repo.candleCallCount
        )
    }
}
