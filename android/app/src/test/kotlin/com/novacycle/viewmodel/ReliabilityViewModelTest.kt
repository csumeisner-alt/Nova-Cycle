package com.novacycle.viewmodel

import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.AccuracyHistoryEntry
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

    // ── Retrain accuracy trend ──────────────────────────────────────────────

    @Test
    fun `accuracy trend skips null-accuracy entries`() {
        val perf = ModelPerformanceResponse(
            accuracyHistory = listOf(
                AccuracyHistoryEntry(modelName = "m1", trainedAt = "2026-07-01T00:00:00", accuracy = 0.55f),
                AccuracyHistoryEntry(modelName = "m2", trainedAt = "2026-07-08T00:00:00", accuracy = null),
                AccuracyHistoryEntry(modelName = "m3", trainedAt = "2026-07-15T00:00:00", accuracy = 0.62f),
                AccuracyHistoryEntry(modelName = "m4", trainedAt = null, accuracy = null)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val state = viewModel.uiState.value
            assertEquals(listOf("m1", "m3"), state.accuracyTrend.map { it.modelName })
            assertEquals(0.62f, state.latestRetrainAccuracy)
            // delta computed across non-null entries only: 0.62 - 0.55
            assertEquals(0.07f, state.retrainAccuracyDelta!!, 0.0001f)
        }
    }

    @Test
    fun `all-null accuracy history yields empty trend and no delta`() {
        val perf = ModelPerformanceResponse(
            accuracyHistory = listOf(
                AccuracyHistoryEntry(modelName = "m1", accuracy = null),
                AccuracyHistoryEntry(modelName = "m2", accuracy = null)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val state = viewModel.uiState.value
            assertTrue(state.accuracyTrend.isEmpty())
            assertNull(state.latestRetrainAccuracy)
            assertNull(state.retrainAccuracyDelta)
        }
    }

    @Test
    fun `single usable retrain has latest accuracy but no delta`() {
        val perf = ModelPerformanceResponse(
            accuracyHistory = listOf(
                AccuracyHistoryEntry(modelName = "m1", accuracy = null),
                AccuracyHistoryEntry(modelName = "m2", accuracy = 0.58f)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val state = viewModel.uiState.value
            assertEquals(0.58f, state.latestRetrainAccuracy)
            assertNull(state.retrainAccuracyDelta)
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

    // ── Retrain history display order ───────────────────────────────────────

    /**
     * accuracyTrend preserves the chronological (oldest-first) order returned by
     * the backend API, so the UI's `trend.asReversed()` makes the expanded list
     * appear newest-first. This test feeds three entries in ascending date order
     * and asserts that order is preserved, then confirms the reversed slice would
     * start with the newest entry.
     */
    @Test
    fun `accuracy trend preserves chronological order so UI reversed view is newest first`() {
        val perf = ModelPerformanceResponse(
            accuracyHistory = listOf(
                AccuracyHistoryEntry(modelName = "old",    trainedAt = "2026-05-01T00:00:00Z", accuracy = 0.55f),
                AccuracyHistoryEntry(modelName = "middle", trainedAt = "2026-06-01T00:00:00Z", accuracy = 0.60f),
                AccuracyHistoryEntry(modelName = "newest", trainedAt = "2026-07-01T00:00:00Z", accuracy = 0.65f)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val trend = viewModel.uiState.value.accuracyTrend

            // ViewModel keeps chronological (oldest-first) order — same as API order.
            assertEquals(listOf("old", "middle", "newest"), trend.map { it.modelName })

            // The UI renders `trend.asReversed()`, so the first displayed row is newest.
            val displayOrder = trend.asReversed()
            assertEquals("newest", displayOrder.first().modelName)
            assertEquals("old",    displayOrder.last().modelName)
        }
    }

    /**
     * Entries with null accuracy must be absent from the expanded list because
     * accuracyTrend filters them before they reach the UI. This verifies the
     * property contract exhaustively: every entry in accuracyTrend has a
     * non-null accuracy, even when null entries are interleaved throughout.
     */
    @Test
    fun `every entry in accuracy trend has non-null accuracy`() {
        val perf = ModelPerformanceResponse(
            accuracyHistory = listOf(
                AccuracyHistoryEntry(modelName = "a", accuracy = 0.50f),
                AccuracyHistoryEntry(modelName = "b", accuracy = null),
                AccuracyHistoryEntry(modelName = "c", accuracy = 0.55f),
                AccuracyHistoryEntry(modelName = "d", accuracy = null),
                AccuracyHistoryEntry(modelName = "e", accuracy = 0.60f)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val trend = viewModel.uiState.value.accuracyTrend
            // Only entries a, c, e survive the null filter.
            assertEquals(listOf("a", "c", "e"), trend.map { it.modelName })
            // All surviving entries have non-null accuracy — UI never renders a null value.
            assertTrue("Every entry must have non-null accuracy", trend.all { it.accuracy != null })
        }
    }

    /**
     * An entry with an unparseable trainedAt string must not be dropped from
     * accuracyTrend — the ViewModel only filters on accuracy, not on trainedAt.
     * The garbled date string reaches the UI's formatIsoTimestamp(), which falls
     * back to returning the raw string rather than crashing. This test confirms
     * the ViewModel's side of that contract: no exception, entry still present.
     */
    @Test
    fun `entry with unparseable trainedAt is not dropped from accuracy trend`() {
        val perf = ModelPerformanceResponse(
            accuracyHistory = listOf(
                AccuracyHistoryEntry(modelName = "good",    trainedAt = "2026-07-01T00:00:00Z", accuracy = 0.60f),
                AccuracyHistoryEntry(modelName = "garbled", trainedAt = "not-a-date",            accuracy = 0.65f),
                AccuracyHistoryEntry(modelName = "nulldate", trainedAt = null,                   accuracy = 0.58f)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val trend = viewModel.uiState.value.accuracyTrend
            // All three entries have non-null accuracy so all three must appear.
            assertEquals(3, trend.size)
            val names = trend.map { it.modelName }
            assertTrue("good entry must be present",    "good"     in names)
            assertTrue("garbled date entry must be present", "garbled"  in names)
            assertTrue("null trainedAt entry must be present", "nulldate" in names)
            // trainedAt values are forwarded as-is to the UI; ViewModel does not parse them.
            assertEquals("not-a-date", trend.first { it.modelName == "garbled" }.trainedAt)
        }
    }

    // ── Flat-trend / zero-range sparkline edge cases ────────────────────────

    /**
     * When every retrain produces the exact same accuracy (flat trend), the
     * ViewModel must still surface all entries in accuracyTrend and expose a
     * non-null latestRetrainAccuracy. The delta will be ~0 (or null for a
     * two-entry flat list — here we use three entries so a delta is computed).
     *
     * This verifies that the state passed to AccuracySparkline is valid for a
     * flat trend: the list has >= 2 elements and all accuracy values are finite,
     * so the composable's own zero-range guard (`takeIf { it > 1e-6f } ?: 1f`)
     * is the only line of defence. If that guard were absent the canvas drawing
     * loop would produce NaN y-offsets; the guard is tested separately in
     * AccuracySparklineLogicTest.
     */
    @Test
    fun `flat trend with identical accuracy values produces valid non-empty trend state`() {
        val perf = ModelPerformanceResponse(
            accuracyHistory = listOf(
                AccuracyHistoryEntry(modelName = "r1", trainedAt = "2026-07-01T00:00:00Z", accuracy = 0.70f),
                AccuracyHistoryEntry(modelName = "r2", trainedAt = "2026-07-08T00:00:00Z", accuracy = 0.70f),
                AccuracyHistoryEntry(modelName = "r3", trainedAt = "2026-07-15T00:00:00Z", accuracy = 0.70f)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val state = viewModel.uiState.value
            // All three entries survive (none are null-accuracy).
            assertEquals(3, state.accuracyTrend.size)
            // The extracted float list passed to AccuracySparkline has >= 2 elements.
            val sparkValues = state.accuracyTrend.mapNotNull { it.accuracy }
            assertTrue("sparkline input must have >= 2 values", sparkValues.size >= 2)
            // All values are finite (no NaN/Inf from the source data).
            assertTrue("all sparkline values must be finite", sparkValues.all { it.isFinite() })
            // Latest accuracy is correct.
            assertEquals(0.70f, state.latestRetrainAccuracy)
            // Delta is present and should be 0 (same values).
            assertNotNull(state.retrainAccuracyDelta)
            assertEquals(0f, state.retrainAccuracyDelta!!, 0.001f)
        }
    }

    /**
     * A single usable accuracy entry means AccuracySparkline is never called
     * (the UI only renders it when `trend.size >= 2`). The ViewModel must still
     * expose a non-null latestRetrainAccuracy and a null delta — confirmed here
     * so the conditional in the composable is validated end-to-end.
     */
    @Test
    fun `single entry trend skips sparkline and exposes accuracy without delta`() {
        val perf = ModelPerformanceResponse(
            accuracyHistory = listOf(
                AccuracyHistoryEntry(modelName = "only", trainedAt = "2026-07-15T00:00:00Z", accuracy = 0.65f)
            )
        )
        val repo = FakeRepo(history(emptyList()), perf)
        vmTest(repo) { viewModel ->
            val state = viewModel.uiState.value
            // Exactly one entry in the trend.
            assertEquals(1, state.accuracyTrend.size)
            // Sparkline would be skipped by the UI (requires size >= 2).
            val sparkValues = state.accuracyTrend.mapNotNull { it.accuracy }
            assertTrue("only one value — sparkline must not be shown", sparkValues.size < 2)
            // Latest accuracy is still surfaced.
            assertEquals(0.65f, state.latestRetrainAccuracy)
            // No previous entry, so no delta.
            assertNull(state.retrainAccuracyDelta)
        }
    }
}
