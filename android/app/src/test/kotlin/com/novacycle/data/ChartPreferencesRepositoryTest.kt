package com.novacycle.data

import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import com.novacycle.data.repository.ChartPreferencesRepository
import com.novacycle.data.repository.ChartScreenKey
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Tests for ChartPreferencesRepository:
 *  - defaults when nothing is stored
 *  - timeframe + render mode round-trip per screen
 *  - RAW and FILTERED screens store independent values
 *  - unknown stored values fall back to defaults
 *  - invalid values passed to save are rejected (not persisted)
 */
class ChartPreferencesRepositoryTest {

    private lateinit var file: File
    private lateinit var scopeJob: Job
    private lateinit var repo: ChartPreferencesRepository
    private lateinit var store: androidx.datastore.core.DataStore<androidx.datastore.preferences.core.Preferences>

    @Before
    fun setUp() {
        file = File.createTempFile("chart_prefs_test", ".preferences_pb")
        file.delete()
        scopeJob = Job()
        store = PreferenceDataStoreFactory.create(
            scope = CoroutineScope(Dispatchers.IO + scopeJob)
        ) { file }
        repo = ChartPreferencesRepository(store)
    }

    @After
    fun tearDown() = runBlocking {
        scopeJob.cancelAndJoin()
        file.delete()
        Unit
    }

    @Test
    fun `defaults when nothing stored`() = runBlocking {
        val prefs = repo.prefs(ChartScreenKey.RAW).first()
        assertEquals("daily", prefs.timeframe)
        assertEquals("CANDLES", prefs.renderMode)
    }

    @Test
    fun `timeframe and render mode round-trip`() = runBlocking {
        repo.saveTimeframe(ChartScreenKey.RAW, "15min")
        repo.saveRenderMode(ChartScreenKey.RAW, "LINE")
        val prefs = repo.prefs(ChartScreenKey.RAW).first()
        assertEquals("15min", prefs.timeframe)
        assertEquals("LINE", prefs.renderMode)
    }

    @Test
    fun `screens store independent values`() = runBlocking {
        repo.saveTimeframe(ChartScreenKey.RAW, "5min")
        repo.saveTimeframe(ChartScreenKey.FILTERED, "1h")
        assertEquals("5min", repo.prefs(ChartScreenKey.RAW).first().timeframe)
        assertEquals("1h", repo.prefs(ChartScreenKey.FILTERED).first().timeframe)
    }

    @Test
    fun `unknown stored values fall back to defaults`() = runBlocking {
        store.edit {
            it[stringPreferencesKey("rawChartTimeframe")] = "2min"
            it[stringPreferencesKey("rawChartRenderMode")] = "HEIKIN_ASHI"
        }
        val prefs = repo.prefs(ChartScreenKey.RAW).first()
        assertEquals("daily", prefs.timeframe)
        assertEquals("CANDLES", prefs.renderMode)
    }

    @Test
    fun `invalid save values are rejected`() = runBlocking {
        repo.saveTimeframe(ChartScreenKey.RAW, "15min")
        repo.saveTimeframe(ChartScreenKey.RAW, "bogus")
        repo.saveRenderMode(ChartScreenKey.RAW, "bogus")
        val prefs = repo.prefs(ChartScreenKey.RAW).first()
        assertEquals("15min", prefs.timeframe)
        assertEquals("CANDLES", prefs.renderMode)
    }
}
