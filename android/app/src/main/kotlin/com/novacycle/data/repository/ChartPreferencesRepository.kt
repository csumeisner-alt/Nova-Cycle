package com.novacycle.data.repository

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.stringPreferencesKey
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

/** Which chart screen the preference belongs to. */
enum class ChartScreenKey(val prefix: String) {
    RAW("rawChart"),
    FILTERED("filteredChart")
}

/** Persisted chart display preferences for one screen. */
data class ChartPrefs(
    /** API timeframe value: 'daily', '5min', '15min' or '1h' */
    val timeframe: String = "daily",
    /** Render mode name: 'CANDLES' or 'LINE' */
    val renderMode: String = "CANDLES"
) {
    companion object {
        val VALID_TIMEFRAMES = setOf("daily", "5min", "15min", "1h")
        val VALID_RENDER_MODES = setOf("CANDLES", "LINE")
    }
}

/**
 * DataStore-backed persistence for per-screen chart preferences (selected
 * timeframe and candles/line render mode), so traders who live on the 15m
 * chart don't have to re-select it on every launch.
 *
 * Unknown/corrupt stored values fall back to defaults rather than propagating
 * an invalid timeframe into API calls.
 */
@Singleton
open class ChartPreferencesRepository @Inject constructor(
    private val dataStore: DataStore<Preferences>
) {
    private fun timeframeKey(screen: ChartScreenKey) =
        stringPreferencesKey("${screen.prefix}Timeframe")

    private fun renderModeKey(screen: ChartScreenKey) =
        stringPreferencesKey("${screen.prefix}RenderMode")

    open fun prefs(screen: ChartScreenKey): Flow<ChartPrefs> = dataStore.data
        .catch { emit(emptyPreferences()) }
        .map { p ->
            ChartPrefs(
                timeframe = p[timeframeKey(screen)]
                    ?.takeIf { it in ChartPrefs.VALID_TIMEFRAMES } ?: "daily",
                renderMode = p[renderModeKey(screen)]
                    ?.takeIf { it in ChartPrefs.VALID_RENDER_MODES } ?: "CANDLES"
            )
        }

    open suspend fun saveTimeframe(screen: ChartScreenKey, timeframe: String) {
        if (timeframe !in ChartPrefs.VALID_TIMEFRAMES) return
        dataStore.edit { it[timeframeKey(screen)] = timeframe }
    }

    open suspend fun saveRenderMode(screen: ChartScreenKey, renderMode: String) {
        if (renderMode !in ChartPrefs.VALID_RENDER_MODES) return
        dataStore.edit { it[renderModeKey(screen)] = renderMode }
    }
}
