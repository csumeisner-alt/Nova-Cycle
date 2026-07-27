package com.novacycle.data.repository

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import com.novacycle.ui.theme.NovaTheme
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * JVM tests for the theme persistence + unlock milestone logic, backed by a real
 * Preferences DataStore writing to a temp file.
 */
class ThemeRepositoryTest {

    private lateinit var job: Job
    private lateinit var scope: CoroutineScope
    private lateinit var file: File
    private lateinit var dataStore: DataStore<Preferences>
    private lateinit var repo: ThemeRepository

    @Before
    fun setUp() {
        job = Job()
        scope = CoroutineScope(Dispatchers.IO + job)
        file = File.createTempFile("theme-test", ".preferences_pb").also { it.delete() }
        dataStore = PreferenceDataStoreFactory.create(scope = scope) { file }
        repo = ThemeRepository(dataStore)
    }

    @After
    fun tearDown() {
        scope.cancel()
        file.delete()
    }

    @Test
    fun `defaults - DarkLuxe selected, only free themes unlocked`() = runBlocking {
        val state = repo.themeState.first()
        assertEquals(NovaTheme.DARK_LUXE, state.selectedTheme)
        assertEquals(0L, state.tapCount)
        assertTrue(state.isUnlocked(NovaTheme.DARK_LUXE))
        assertTrue(state.isUnlocked(NovaTheme.MINT_LUXE))
        assertFalse(state.isUnlocked(NovaTheme.AURORA_FLUX))
        assertFalse(state.isUnlocked(NovaTheme.CRIMSON_PULSE))
    }

    @Test
    fun `taps accumulate across batches`() = runBlocking {
        repo.addTaps(100)
        repo.addTaps(250)
        assertEquals(350L, repo.themeState.first().tapCount)
    }

    @Test
    fun `aurora unlocks exactly at 10000 taps`() = runBlocking {
        repo.addTaps(9_999)
        assertFalse(repo.themeState.first().isUnlocked(NovaTheme.AURORA_FLUX))
        val unlocked = repo.addTaps(1)
        assertEquals(listOf(NovaTheme.AURORA_FLUX), unlocked)
        assertTrue(repo.themeState.first().isUnlocked(NovaTheme.AURORA_FLUX))
    }

    @Test
    fun `unlock is reported exactly once`() = runBlocking {
        assertEquals(listOf(NovaTheme.AURORA_FLUX), repo.addTaps(15_000))
        // Further taps below the next milestone report nothing new
        assertEquals(emptyList<NovaTheme>(), repo.addTaps(100))
    }

    @Test
    fun `single big batch can unlock both themes at once`() = runBlocking {
        val unlocked = repo.addTaps(25_000)
        assertEquals(listOf(NovaTheme.AURORA_FLUX, NovaTheme.CRIMSON_PULSE), unlocked)
        val state = repo.themeState.first()
        assertTrue(state.isUnlocked(NovaTheme.AURORA_FLUX))
        assertTrue(state.isUnlocked(NovaTheme.CRIMSON_PULSE))
    }

    @Test
    fun `crimson unlocks at 20000 taps`() = runBlocking {
        repo.addTaps(19_999)
        assertFalse(repo.themeState.first().isUnlocked(NovaTheme.CRIMSON_PULSE))
        assertEquals(listOf(NovaTheme.CRIMSON_PULSE), repo.addTaps(1))
    }

    @Test
    fun `selecting an unlocked theme persists`() = runBlocking {
        assertTrue(repo.selectTheme(NovaTheme.MINT_LUXE))
        assertEquals(NovaTheme.MINT_LUXE, repo.themeState.first().selectedTheme)
    }

    @Test
    fun `selecting a locked theme is rejected`() = runBlocking {
        assertFalse(repo.selectTheme(NovaTheme.CRIMSON_PULSE))
        assertEquals(NovaTheme.DARK_LUXE, repo.themeState.first().selectedTheme)
    }

    @Test
    fun `locked theme becomes selectable after unlock`() = runBlocking {
        repo.addTaps(10_000)
        assertTrue(repo.selectTheme(NovaTheme.AURORA_FLUX))
        assertEquals(NovaTheme.AURORA_FLUX, repo.themeState.first().selectedTheme)
    }

    @Test
    fun `zero or negative tap batches are no-ops`() = runBlocking {
        assertEquals(emptyList<NovaTheme>(), repo.addTaps(0))
        assertEquals(emptyList<NovaTheme>(), repo.addTaps(-5))
        assertEquals(0L, repo.themeState.first().tapCount)
    }

    @Test
    fun `state survives reopening the store`() = runBlocking {
        repo.addTaps(10_500)
        repo.selectTheme(NovaTheme.AURORA_FLUX)
        // Simulate app restart: fully shut down the old DataStore (cancelAndJoin —
        // DataStore releases its single-instance file guard asynchronously, and a
        // second instance over a still-guarded file fails its reads), then reopen.
        job.cancelAndJoin()
        job = Job()
        scope = CoroutineScope(Dispatchers.IO + job)
        dataStore = PreferenceDataStoreFactory.create(scope = scope) { file }
        repo = ThemeRepository(dataStore)
        val state = repo.themeState.first()
        assertEquals(NovaTheme.AURORA_FLUX, state.selectedTheme)
        assertEquals(10_500L, state.tapCount)
        assertTrue(state.isUnlocked(NovaTheme.AURORA_FLUX))
        assertFalse(state.isUnlocked(NovaTheme.CRIMSON_PULSE))
    }
}
