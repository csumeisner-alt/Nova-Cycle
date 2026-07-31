package com.novacycle.viewmodel

import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.CalibrationPoint
import com.novacycle.data.remote.models.ConfidenceBucket
import com.novacycle.data.remote.models.ModelPerformanceResponse
import com.novacycle.data.remote.models.ModelPerformanceSummary
import com.novacycle.data.remote.models.MissedRallies
import com.novacycle.data.remote.models.ReliabilityMetricsResponse
import com.novacycle.data.remote.models.TradeCycleResponse
import com.novacycle.data.remote.models.TradeHistoryResponse
import com.novacycle.data.repository.DataFreshnessTracker
import com.novacycle.data.repository.NovaCycleRepository
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Tests for the model-performance additions on [ReliabilityViewModel]:
 *   - period / confidence chips update the outgoing query parameters
 *   - an empty API response yields an empty state without crashing
 *   - the HIGH confidence band filters cycles to confidenceAtBuy in [0.7, 1.0]
 *   - cumulative P&L exposes the raw percent for "+" / "−" formatting
 *
 * Uses a fake NovaCycleRepository (real subclass) so the suspend funs returning
 * kotlin.Result behave correctly under a StandardTestDispatcher.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ReliabilityViewModelTest {

    private val dispatcher = StandardTestDispatcher()

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    // ── Fakes ────────────────────────────────────────────────────────────────

    private fun cycle(id: String, confidence: Float?, returnPercent: Float): TradeCycleResponse =
        TradeCycleResponse(
            cycleId = id,
            confidenceAtBuy = confidence,
            returnPercent = returnPercent
        )

    private fun history(cycles: List<TradeCycleResponse>) = TradeHistoryResponse(
        ticker = "VOO",
        cycles = cycles,
        summary = ReliabilityMetricsResponse()
    )

    /**
     * Fake repository that records the query parameters it was called with and
     * returns caller-supplied fixtures. Constructor collaborators are inert mocks.
     */
    private class FakeRepo(
        private val historyResponse: TradeHistoryResponse,
        private val performanceResponse: ModelPerformanceResponse
    ) : NovaCycleRepository(
        apiService = mockk(),
        signalDao = mockk(),
        confidenceDao = mockk(),
        candleDao = mockk(),
        freshnessTracker = DataFreshnessTracker()
    ) {
        var lastHistoryWindow: String? = null
        var lastPerfWindow: String? = null
        var lastPerfMin: Float? = null
        var lastPerfMax: Float? = null

        override suspend fun getTradeHistory(
            ticker: String,
            window: String
        ): Result<TradeHistoryResponse> {
            lastHistoryWindow = window
            return Result.success(historyResponse)
        }

        override suspend fun getModelPerformance(
            ticker: String,
            window: String,
            period: String,
            confidenceMin: Float?,
            confidenceMax: Float?
        ): Result<ModelPerformanceResponse> {
            lastPerfWindow = window
            lastPerfMin = confidenceMin
            lastPerfMax = confidenceMax
            return Result.success(performanceResponse)
        }
    }

    private fun vmTest(
        repo: FakeRepo,
        block: TestScope.(ReliabilityViewModel) -> Unit
    ) = runTest(dispatcher) {
        val viewModel = ReliabilityViewModel(repo)
        try {
            runCurrent() // let init's loadTradeHistory + loadModelPerformance settle
            block(viewModel)
        } finally {
            viewModel.viewModelScope.cancel()
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────────────

    @Test
    fun `default load uses 30d window`() {
        val repo = FakeRepo(history(emptyList()), ModelPerformanceResponse())
        vmTest(repo) {
            assertEquals("30d", repo.lastHistoryWindow)
            assertEquals("30d", repo.lastPerfWindow)
            assertNull(repo.lastPerfMin)
            assertNull(repo.lastPerfMax)
        }
    }

    @Test
    fun `period chip updates query window for both endpoints`() {
        val repo = FakeRepo(history(emptyList()), ModelPerformanceResponse())
        vmTest(repo) { viewModel ->
            viewModel.setPeriodFilter(PeriodFilter.D1)
            runCurrent()
            assertEquals("1d", repo.lastHistoryWindow)
            assertEquals("1d", repo.lastPerfWindow)

            viewModel.setPeriodFilter(PeriodFilter.D7)
            runCurrent()
            assertEquals("7d", repo.lastHistoryWindow)
            assertEquals("7d", repo.lastPerfWindow)
        }
    }

    @Test
    fun `confidence chip updates confidence_min and confidence_max`() {
        val repo = FakeRepo(history(emptyList()), ModelPerformanceResponse())
        vmTest(repo) { viewModel ->
            viewModel.setConfidenceBand(ConfidenceBand.HIGH)
            runCurrent()
            assertEquals(0.7f, repo.lastPerfMin)
            assertEquals(1.0f, repo.lastPerfMax)

            viewModel.setConfidenceBand(ConfidenceBand.LOW)
            runCurrent()
            assertEquals(0.0f, repo.lastPerfMin)
            assertEquals(0.4f, repo.lastPerfMax)

            viewModel.setConfidenceBand(ConfidenceBand.ALL)
            runCurrent()
            assertNull(repo.lastPerfMin)
            assertNull(repo.lastPerfMax)
        }
    }

    @Test
    fun `empty API response yields empty state without crashing`() {
        val repo = FakeRepo(history(emptyList()), ModelPerformanceResponse())
        vmTest(repo) { viewModel ->
            val state = viewModel.uiState.value
            assertTrue(state.cycles.isEmpty())
            assertTrue(state.filteredCycles.isEmpty())
            assertEquals(0, state.missedRallyCount)
            assertEquals(0f, state.cumulativeReturnPercent)
            assertNull(state.error)
            assertNull(state.performanceError)
            assertNotNull(state.performance)
        }
    }

    @Test
    fun `HIGH band filters cycles by confidenceAtBuy in half-open range including top`() {
        val cycles = listOf(
            cycle("low", 0.2f, 1f),
            cycle("mid", 0.5f, 1f),
            cycle("edge-low", 0.7f, 1f),  // included (min inclusive)
            cycle("high", 0.9f, 1f),      // included
            cycle("edge-high", 1.0f, 1f), // included (top of scale)
            cycle("null", null, 1f)       // excluded (no confidence)
        )
        val repo = FakeRepo(history(cycles), ModelPerformanceResponse())
        vmTest(repo) { viewModel ->
            viewModel.setConfidenceBand(ConfidenceBand.HIGH)
            runCurrent()
            val ids = viewModel.uiState.value.filteredCycles.map { it.cycleId }.toSet()
            assertEquals(setOf("edge-low", "high", "edge-high"), ids)
        }
    }

    @Test
    fun `MEDIUM band is half-open excluding upper bound`() {
        val cycles = listOf(
            cycle("below", 0.39f, 1f),       // excluded
            cycle("edge-low", 0.4f, 1f),     // included (min inclusive) — MEDIUM not LOW
            cycle("inside", 0.55f, 1f),      // included
            cycle("edge-high", 0.7f, 1f),    // excluded (max exclusive) — belongs to HIGH
            cycle("above", 0.8f, 1f)         // excluded
        )
        val repo = FakeRepo(history(cycles), ModelPerformanceResponse())
        vmTest(repo) { viewModel ->
            viewModel.setConfidenceBand(ConfidenceBand.MEDIUM)
            runCurrent()
            val ids = viewModel.uiState.value.filteredCycles.map { it.cycleId }.toSet()
            assertEquals(setOf("edge-low", "inside"), ids)
        }
    }

    @Test
    fun `LOW band is half-open excluding upper bound`() {
        val cycles = listOf(
            cycle("zero", 0.0f, 1f),      // included (min inclusive)
            cycle("inside", 0.2f, 1f),    // included
            cycle("edge-high", 0.4f, 1f), // excluded (max exclusive) — belongs to MEDIUM
            cycle("above", 0.5f, 1f)      // excluded
        )
        val repo = FakeRepo(history(cycles), ModelPerformanceResponse())
        vmTest(repo) { viewModel ->
            viewModel.setConfidenceBand(ConfidenceBand.LOW)
            runCurrent()
            val ids = viewModel.uiState.value.filteredCycles.map { it.cycleId }.toSet()
            assertEquals(setOf("zero", "inside"), ids)
        }
    }

    @Test
    fun `cumulative P&L positive value formats with plus`() {
        val perf = ModelPerformanceResponse(
            summary = ModelPerformanceSummary(cumulativeReturnPercent = 3.25f)
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val pct = viewModel.uiState.value.cumulativeReturnPercent
            assertTrue(pct > 0f)
            val prefix = if (pct >= 0f) "+" else "−"
            assertEquals("+", prefix)
        }
    }

    @Test
    fun `cumulative P&L negative value formats with minus`() {
        val perf = ModelPerformanceResponse(
            summary = ModelPerformanceSummary(cumulativeReturnPercent = -1.5f)
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val pct = viewModel.uiState.value.cumulativeReturnPercent
            assertTrue(pct < 0f)
            val prefix = if (pct >= 0f) "+" else "−"
            assertEquals("−", prefix)
        }
    }

    @Test
    fun `high-confidence claim falls back to 85 percent when calibration missing`() {
        val perf = ModelPerformanceResponse(
            confidenceBuckets = mapOf("high" to ConfidenceBucket(tradeCount = 4, winRate = 0.75f)),
            calibrationCurve = emptyList()
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val state = viewModel.uiState.value
            assertEquals(0.75f, state.highConfidenceWinRate)
            assertEquals(0.85f, state.highConfidenceClaim)
        }
    }

    @Test
    fun `high-confidence claim uses calibration midpoints when available`() {
        val perf = ModelPerformanceResponse(
            confidenceBuckets = mapOf("high" to ConfidenceBucket(tradeCount = 4, winRate = 0.8f)),
            calibrationCurve = listOf(
                CalibrationPoint(confidenceMid = 0.75f, actualWinRate = 0.7f, tradeCount = 2),
                CalibrationPoint(confidenceMid = 0.85f, actualWinRate = 0.9f, tradeCount = 2),
                CalibrationPoint(confidenceMid = 0.55f, actualWinRate = 0.5f, tradeCount = 5) // ignored (< 0.7)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            // avg of 0.75 and 0.85 == 0.80
            assertEquals(0.80f, viewModel.uiState.value.highConfidenceClaim, 0.0001f)
        }
    }

    @Test
    fun `missed rally count surfaces from performance feed`() {
        val perf = ModelPerformanceResponse(
            missedRallies = MissedRallies(count = 5, rate = 0.25f)
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            assertEquals(5, viewModel.uiState.value.missedRallyCount)
        }
    }
}
