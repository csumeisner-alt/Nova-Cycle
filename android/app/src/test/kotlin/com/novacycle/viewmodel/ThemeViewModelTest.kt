package com.novacycle.viewmodel

import androidx.lifecycle.viewModelScope
import com.novacycle.data.repository.ThemeRepository
import com.novacycle.ui.theme.NovaTheme
import io.mockk.mockk
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

/**
 * Tests for the ThemeViewModel tap-batching flush loop:
 *
 *  - taps registered between ticks are flushed in a single batch (no drops)
 *  - taps accumulated across several intervals are all delivered
 *  - flushNow() persists pending taps immediately (the onStop lifecycle hook)
 *  - concurrent registerTap() from many threads never loses a tap (AtomicInteger)
 *  - unlock events surfaced by the repository are re-emitted exactly once
 *  - empty intervals do not call the repository at all
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ThemeViewModelTest {

    private val dispatcher = StandardTestDispatcher()

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    /**
     * Fake repository recording each addTaps batch. Optionally reports themes
     * as newly unlocked on specific calls. The DataStore collaborator is an
     * inert mock — only the overridden members are exercised.
     */
    private class FakeThemeRepository(
        private val unlocksByCall: Map<Int, List<NovaTheme>> = emptyMap()
    ) : ThemeRepository(dataStore = mockk(relaxed = true)) {
        val batches = mutableListOf<Int>()
        override val themeState = MutableStateFlow(com.novacycle.data.repository.ThemeState())

        override suspend fun addTaps(count: Int): List<NovaTheme> {
            batches.add(count)
            return unlocksByCall[batches.size] ?: emptyList()
        }

        override suspend fun selectTheme(theme: NovaTheme): Boolean = true
    }

    /**
     * Runs [block] against a fresh ThemeViewModel + fake repo, always cancelling
     * viewModelScope in finally — otherwise the infinite flush loop keeps
     * advancing virtual time forever during runTest cleanup.
     */
    private fun themeTest(
        repo: FakeThemeRepository = FakeThemeRepository(),
        block: TestScope.(ThemeViewModel, FakeThemeRepository) -> Unit
    ) = runTest(dispatcher) {
        val viewModel = ThemeViewModel(repo)
        try {
            block(viewModel, repo)
        } finally {
            viewModel.viewModelScope.cancel()
        }
    }

    /** Advance past exactly one flush interval. */
    private fun TestScope.nextFlush() = advanceTimeBy(ThemeViewModel.TAP_FLUSH_INTERVAL_MS + 1)

    @Test
    fun `taps registered before a tick are flushed as one batch`() = themeTest { viewModel, repo ->
        repeat(7) { viewModel.registerTap() }
        nextFlush()
        assertEquals(listOf(7), repo.batches)
    }

    @Test
    fun `no taps means no repository writes`() = themeTest { viewModel, repo ->
        repeat(5) { nextFlush() }
        assertTrue(repo.batches.isEmpty())
    }

    @Test
    fun `taps across several intervals are all delivered without drops`() = themeTest { viewModel, repo ->
        repeat(3) { viewModel.registerTap() }
        nextFlush()
        repeat(10) { viewModel.registerTap() }
        nextFlush()
        nextFlush() // idle interval — no batch
        viewModel.registerTap()
        nextFlush()
        assertEquals(listOf(3, 10, 1), repo.batches)
        assertEquals(14, repo.batches.sum())
    }

    @Test
    fun `flushNow persists pending taps immediately without waiting for the tick`() =
        themeTest { viewModel, repo ->
            repeat(4) { viewModel.registerTap() }
            viewModel.flushNow()
            runCurrent() // run the launched flush, no time advancement
            assertEquals(listOf(4), repo.batches)
            // The periodic tick afterwards has nothing left to flush.
            nextFlush()
            assertEquals(listOf(4), repo.batches)
        }

    @Test
    fun `flushNow with no pending taps is a no-op`() = themeTest { viewModel, repo ->
        viewModel.flushNow()
        runCurrent()
        assertTrue(repo.batches.isEmpty())
    }

    @Test
    fun `unlock events are re-emitted exactly once per unlock`() = themeTest(
        FakeThemeRepository(unlocksByCall = mapOf(1 to listOf(NovaTheme.AURORA_FLUX)))
    ) { viewModel, repo ->
        val received = mutableListOf<NovaTheme>()
        val collector = launch { viewModel.unlockEvents.collect { received.add(it) } }
        viewModel.registerTap()
        nextFlush()
        viewModel.registerTap()
        nextFlush()
        assertEquals(listOf(NovaTheme.AURORA_FLUX), received)
        collector.cancel()
    }

    @Test
    fun `rapid concurrent taps from many threads are never dropped`() = themeTest { viewModel, repo ->
        // Simulate frantic multi-touch: 8 real threads x 500 taps each hammering
        // registerTap() simultaneously. registerTap must be lock-free & thread-safe.
        val threads = 8
        val tapsPerThread = 500
        val start = CountDownLatch(1)
        val done = CountDownLatch(threads)
        repeat(threads) {
            thread {
                start.await()
                repeat(tapsPerThread) { viewModel.registerTap() }
                done.countDown()
            }
        }
        start.countDown()
        assertTrue("tap threads did not finish", done.await(10, TimeUnit.SECONDS))
        nextFlush()
        assertEquals(threads * tapsPerThread, repo.batches.sum())
    }

    @Test
    fun `taps landing between flushNow and the next tick are still counted`() =
        themeTest { viewModel, repo ->
            repeat(2) { viewModel.registerTap() }
            viewModel.flushNow()
            runCurrent()
            // Taps after the lifecycle flush go into the next periodic batch.
            repeat(3) { viewModel.registerTap() }
            nextFlush()
            assertEquals(listOf(2, 3), repo.batches)
        }
}
