package com.novacycle.data.repository

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import com.novacycle.ui.theme.NovaTheme
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Snapshot of everything the theme system persists.
 */
data class ThemeState(
    val selectedTheme: NovaTheme = NovaTheme.DEFAULT,
    val tapCount: Long = 0L,
    val unlockedThemes: Set<NovaTheme> = setOf(NovaTheme.DARK_LUXE, NovaTheme.MINT_LUXE)
) {
    fun isUnlocked(theme: NovaTheme): Boolean = theme in unlockedThemes
}

/**
 * DataStore-backed persistence for the luxe theme system: cumulative tap count,
 * per-theme unlock flags, and the selected theme.
 *
 * Unlock flags are written in the SAME DataStore transaction that crosses the
 * milestone, so an unlock can never be lost between "count says unlocked" and
 * "flag says unlocked". [addTaps] reports themes whose flag transitioned
 * false → true in that transaction — the exactly-once unlock celebration signal.
 */
@Singleton
open class ThemeRepository @Inject constructor(
    private val dataStore: DataStore<Preferences>
) {
    companion object {
        val KEY_TAP_COUNT        = longPreferencesKey("tapCount")
        val KEY_AURORA_UNLOCKED  = booleanPreferencesKey("auroraUnlocked")
        val KEY_CRIMSON_UNLOCKED = booleanPreferencesKey("crimsonUnlocked")
        val KEY_MINT_UNLOCKED    = booleanPreferencesKey("mintUnlocked")
        val KEY_SELECTED_THEME   = stringPreferencesKey("selectedTheme")
    }

    open val themeState: Flow<ThemeState> = dataStore.data
        .catch { emit(emptyPreferences()) }
        .map { prefs -> prefs.toThemeState() }

    /**
     * Add [count] taps to the cumulative total. Returns the list of themes that
     * became newly unlocked by this batch (usually empty).
     */
    open suspend fun addTaps(count: Int): List<NovaTheme> {
        if (count <= 0) return emptyList()
        val newlyUnlocked = mutableListOf<NovaTheme>()
        dataStore.edit { prefs ->
            val total = (prefs[KEY_TAP_COUNT] ?: 0L) + count
            prefs[KEY_TAP_COUNT] = total
            if (prefs[KEY_MINT_UNLOCKED] != true) prefs[KEY_MINT_UNLOCKED] = true
            if (total >= NovaTheme.AURORA_FLUX.unlockTaps && prefs[KEY_AURORA_UNLOCKED] != true) {
                prefs[KEY_AURORA_UNLOCKED] = true
                newlyUnlocked += NovaTheme.AURORA_FLUX
            }
            if (total >= NovaTheme.CRIMSON_PULSE.unlockTaps && prefs[KEY_CRIMSON_UNLOCKED] != true) {
                prefs[KEY_CRIMSON_UNLOCKED] = true
                newlyUnlocked += NovaTheme.CRIMSON_PULSE
            }
        }
        return newlyUnlocked
    }

    /**
     * Persist the selected theme. Locked themes are rejected (returns false) —
     * the UI greys them out, but this is the authoritative guard.
     */
    open suspend fun selectTheme(theme: NovaTheme): Boolean {
        var accepted = false
        dataStore.edit { prefs ->
            if (prefs.toThemeState().isUnlocked(theme)) {
                prefs[KEY_SELECTED_THEME] = theme.storageKey
                accepted = true
            }
        }
        return accepted
    }

    private fun Preferences.toThemeState(): ThemeState {
        val unlocked = buildSet {
            add(NovaTheme.DARK_LUXE)
            add(NovaTheme.MINT_LUXE) // always unlocked regardless of flag
            if (this@toThemeState[KEY_AURORA_UNLOCKED] == true) add(NovaTheme.AURORA_FLUX)
            if (this@toThemeState[KEY_CRIMSON_UNLOCKED] == true) add(NovaTheme.CRIMSON_PULSE)
        }
        // A persisted selection pointing at a theme that is somehow locked
        // (e.g. data cleared partially) falls back to the default theme.
        val selected = NovaTheme.fromStorageKey(this[KEY_SELECTED_THEME])
            .takeIf { it in unlocked } ?: NovaTheme.DEFAULT
        return ThemeState(
            selectedTheme = selected,
            tapCount = this[KEY_TAP_COUNT] ?: 0L,
            unlockedThemes = unlocked
        )
    }
}
