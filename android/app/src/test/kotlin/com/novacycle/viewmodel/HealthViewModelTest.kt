package com.novacycle.viewmodel

import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.HealthzResponse
import com.novacycle.data.remote.models.ModelHealth
import com.novacycle.data.repository.DataFreshnessTracker
import com.novacycle.data.repository.NovaCycleRepository
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * State-machine tests for the shared backend-health poll that drives the
 * warning banners on every data screen:
 *
 *  - healthy poll  -> health populated, no degraded/unreachable banner state
 *  - degraded poll -> health populated with isDegraded (amber banner)
 *  - 3 consecutive failed polls -> backendUnreachable = true (distinct notice)
 *  - any successful poll -> clears the unreachable notice and failure count
 *
 * Uses a fake NovaCycleRepository and a coroutine test dispatcher so the
 * 60-second poll loop is driven by virtual time.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class HealthViewModelTest {

    private val dispatcher = StandardTestDispatcher()

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private val healthyResponse = HealthzResponse(status = "ok")

    private val degradedResponse = HealthzResponse(
        status = "degraded",
        models = mapOf(
            "long" to ModelHealth(lastTrainingSuccess = false, lastTrainingError = "retrain failed"),
            "short" to ModelHealth(lastTrainingSuccess = true)
        ),
        alerts = listOf("long model failed its last retrain")
    )

    /**
     * Fake repository whose successive getHealth() calls return [results] in
     * order, repeating the last one. Constructor collaborators are inert mocks —
     * only the overridden getHealth() is exercised by HealthViewModel.
     */
    private class FakeNovaCycleRepository(
        private val results: List<Result<HealthzResponse>>
    ) : NovaCycleRepository(
        apiService = mockk(),
        signalDao = mockk(),
        confidenceDao = mockk(),
        candleDao = mockk(),
        freshnessTracker = DataFreshnessTracker()
    ) {
        private var call = 0
        override suspend fun getHealth(): Result<HealthzResponse> {
            val r = results[minOf(call, results.size - 1)]
            call++
            return r
        }
    }

    /**
     * Runs [block] against a fresh HealthViewModel backed by a fake repository
     * returning [results] in order. Always cancels the ViewModel's poll loop
     * afterwards (even on assertion failure) — otherwise the infinite 60s poll
     * loop keeps advancing virtual time forever during runTest cleanup.
     */
    private fun healthTest(
        results: List<Result<HealthzResponse>>,
        block: TestScope.(HealthViewModel) -> Unit
    ) = runTest(dispatcher) {
        val viewModel = HealthViewModel(
            FakeNovaCycleRepository(results),
            DataFreshnessTracker()
        )
        try {
            block(viewModel)
        } finally {
            viewModel.viewModelScope.cancel()
        }
    }

    /** Execute the immediate first poll (fires at t=0, before any delay). */
    private fun TestScope.firstPoll() = runCurrent()

    /** Advance virtual time past one 60s poll interval — exactly one more poll. */
    private fun TestScope.nextPoll() = advanceTimeBy(60_001L)

    @Test
    fun `initial state has no banner`() = healthTest(listOf(Result.success(healthyResponse))) { viewModel ->
        // Before the first poll completes, nothing is shown.
        val state = viewModel.uiState.value
        assertNull(state.health)
        assertFalse(state.backendUnreachable)
    }

    @Test
    fun `healthy poll populates health without any banner state`() =
        healthTest(listOf(Result.success(healthyResponse))) { viewModel ->
            firstPoll()

            val state = viewModel.uiState.value
            assertNotNull(state.health)
            assertFalse(state.health!!.isDegraded)
            assertTrue(state.health!!.degradedModels.isEmpty())
            assertFalse(state.backendUnreachable)
        }

    @Test
    fun `degraded poll populates degraded health state`() =
        healthTest(listOf(Result.success(degradedResponse))) { viewModel ->
            firstPoll()

            val state = viewModel.uiState.value
            assertNotNull(state.health)
            assertTrue(state.health!!.isDegraded)
            assertEquals(listOf("long"), state.health!!.degradedModels)
            assertFalse(state.backendUnreachable)
        }

    @Test
    fun `fewer than three consecutive failures keeps last known health and no unreachable notice`() =
        healthTest(
            listOf(
                Result.success(healthyResponse),
                Result.failure(RuntimeException("timeout")),
                Result.failure(RuntimeException("timeout"))
            )
        ) { viewModel ->
            firstPoll() // success
            nextPoll()  // failure 1
            nextPoll()  // failure 2

            val state = viewModel.uiState.value
            // Transient failures keep the last known health, no unreachable flash.
            assertNotNull(state.health)
            assertFalse(state.backendUnreachable)
        }

    @Test
    fun `three consecutive failures shows backend unreachable`() =
        healthTest(listOf(Result.failure(RuntimeException("connection refused")))) { viewModel ->
            firstPoll() // failure 1
            assertFalse(viewModel.uiState.value.backendUnreachable)
            nextPoll() // failure 2
            assertFalse(viewModel.uiState.value.backendUnreachable)
            nextPoll() // failure 3
            assertTrue(viewModel.uiState.value.backendUnreachable)
            nextPoll() // failure 4 — stays unreachable
            assertTrue(viewModel.uiState.value.backendUnreachable)
        }

    @Test
    fun `successful poll after failures clears the unreachable notice`() =
        healthTest(
            listOf(
                Result.failure(RuntimeException("down")),
                Result.failure(RuntimeException("down")),
                Result.failure(RuntimeException("down")),
                Result.success(healthyResponse)
            )
        ) { viewModel ->
            firstPoll() // failure 1
            repeat(2) { nextPoll() } // failures 2 and 3
            assertTrue(viewModel.uiState.value.backendUnreachable)

            nextPoll() // recovery
            val state = viewModel.uiState.value
            assertFalse(state.backendUnreachable)
            assertNotNull(state.health)
            assertFalse(state.health!!.isDegraded)
        }

    @Test
    fun `unreachable backend is re-polled on the fast 5s recovery cadence`() =
        healthTest(
            listOf(
                Result.failure(RuntimeException("down")),
                Result.failure(RuntimeException("down")),
                Result.failure(RuntimeException("down")),
                Result.success(healthyResponse)
            )
        ) { viewModel ->
            firstPoll() // failure 1
            repeat(2) { nextPoll() } // failures 2 and 3 (60s cadence)
            assertTrue(viewModel.uiState.value.backendUnreachable)

            // Now flagged unreachable — the next poll must fire after only ~5s,
            // not the normal 60s.
            advanceTimeBy(HealthViewModel.RECOVERY_POLL_INTERVAL_MS + 1)
            val state = viewModel.uiState.value
            assertFalse(state.backendUnreachable)
            assertNotNull(state.health)
        }

    @Test
    fun `reachable backend is NOT polled on the fast cadence`() =
        healthTest(
            listOf(
                Result.success(healthyResponse),
                Result.success(degradedResponse)
            )
        ) { viewModel ->
            firstPoll() // success #1
            // 5s later nothing should have happened — normal cadence is 60s.
            advanceTimeBy(HealthViewModel.RECOVERY_POLL_INTERVAL_MS + 1)
            assertFalse(viewModel.uiState.value.health!!.isDegraded)

            nextPoll() // 60s cadence — second poll lands (degraded)
            assertTrue(viewModel.uiState.value.health!!.isDegraded)
        }

    @Test
    fun `recovery also resets the consecutive failure counter`() =
        healthTest(
            listOf(
                Result.failure(RuntimeException("down")),
                Result.failure(RuntimeException("down")),
                Result.success(healthyResponse),
                Result.failure(RuntimeException("down")),
                Result.failure(RuntimeException("down"))
            )
        ) { viewModel ->
            firstPoll() // failure 1
            repeat(2) { nextPoll() } // failure 2, then success — counter reset
            assertFalse(viewModel.uiState.value.backendUnreachable)

            nextPoll() // failure 1 after reset
            nextPoll() // failure 2 after reset
            // Only 2 consecutive failures since the reset — still not unreachable.
            assertFalse(viewModel.uiState.value.backendUnreachable)
        }
}
