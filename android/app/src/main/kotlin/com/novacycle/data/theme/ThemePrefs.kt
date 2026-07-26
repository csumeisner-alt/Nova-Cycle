package com.novacycle.data.theme

import android.content.Context
import android.content.SharedPreferences
import com.novacycle.domain.theme.ThemeUnlockLogic
import com.novacycle.ui.theme.AppTheme
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Snapshot of everything the theme system persists.
 */
data class ThemeState(
    val tapCount: Int = 0,
    val auroraUnlocked: Boolean = false,
    val crimsonUnlocked: Boolean = false,
    val mintUnlocked: Boolean = false,
    val selectedTheme: AppTheme = AppTheme.DARK_LUXE
)

/**
 * Persistence for the tap achievement, theme unlock flags, and selected theme.
 *
 * Stored in SharedPreferences `nova_prefs` with keys `tapCount`,
 * `auroraUnlocked`, `crimsonUnlocked`, `mintUnlocked`, `selectedTheme` —
 * these names are part of the product spec; do not rename them or users
 * lose progress/purchases on upgrade. (The app's DataStore-backed
 * sensitivity settings are a separate store and remain untouched.)
 */
@Singleton
class ThemePrefs @Inject constructor(@ApplicationContext context: Context) {

    companion object {
        const val PREFS_NAME = "nova_prefs"
        const val KEY_TAP_COUNT = "tapCount"
        const val KEY_AURORA_UNLOCKED = "auroraUnlocked"
        const val KEY_CRIMSON_UNLOCKED = "crimsonUnlocked"
        const val KEY_MINT_UNLOCKED = "mintUnlocked"
        const val KEY_SELECTED_THEME = "selectedTheme"
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _state = MutableStateFlow(load())
    val state: StateFlow<ThemeState> = _state.asStateFlow()

    private fun load(): ThemeState {
        val aurora = prefs.getBoolean(KEY_AURORA_UNLOCKED, false)
        val crimson = prefs.getBoolean(KEY_CRIMSON_UNLOCKED, false)
        val mint = prefs.getBoolean(KEY_MINT_UNLOCKED, false)
        val selected = ThemeUnlockLogic.sanitizeSelection(
            AppTheme.fromStorageKey(prefs.getString(KEY_SELECTED_THEME, null)),
            aurora, crimson, mint
        )
        return ThemeState(
            tapCount = prefs.getInt(KEY_TAP_COUNT, 0),
            auroraUnlocked = aurora,
            crimsonUnlocked = crimson,
            mintUnlocked = mint,
            selectedTheme = selected
        )
    }

    /**
     * Record one logo tap. Returns true exactly when this tap crosses the
     * 20,000-tap achievement (callers show the celebration dialog then).
     */
    @Synchronized
    fun registerTap(): Boolean {
        val newCount = _state.value.tapCount + 1
        val unlockNow = ThemeUnlockLogic.isUnlockTap(newCount)
        val editor = prefs.edit().putInt(KEY_TAP_COUNT, newCount)
        if (unlockNow) {
            editor.putBoolean(KEY_AURORA_UNLOCKED, true)
                .putBoolean(KEY_CRIMSON_UNLOCKED, true)
        }
        editor.apply()
        _state.value = _state.value.copy(
            tapCount = newCount,
            auroraUnlocked = _state.value.auroraUnlocked || unlockNow,
            crimsonUnlocked = _state.value.crimsonUnlocked || unlockNow
        )
        return unlockNow
    }

    /** Called by billing after a verified & acknowledged Mint Luxe purchase. */
    @Synchronized
    fun setMintUnlocked(unlocked: Boolean) {
        if (_state.value.mintUnlocked == unlocked) return
        prefs.edit().putBoolean(KEY_MINT_UNLOCKED, unlocked).apply()
        var s = _state.value.copy(mintUnlocked = unlocked)
        // If Mint was revoked (refund), never leave the user on a locked theme.
        s = s.copy(
            selectedTheme = ThemeUnlockLogic.sanitizeSelection(
                s.selectedTheme, s.auroraUnlocked, s.crimsonUnlocked, s.mintUnlocked
            )
        )
        if (s.selectedTheme != _state.value.selectedTheme) {
            prefs.edit().putString(KEY_SELECTED_THEME, s.selectedTheme.storageKey).apply()
        }
        _state.value = s
    }

    /** Select a theme; ignored if the theme is still locked. */
    @Synchronized
    fun selectTheme(theme: AppTheme): Boolean {
        val s = _state.value
        if (!ThemeUnlockLogic.isThemeAvailable(theme, s.auroraUnlocked, s.crimsonUnlocked, s.mintUnlocked)) {
            return false
        }
        prefs.edit().putString(KEY_SELECTED_THEME, theme.storageKey).apply()
        _state.value = s.copy(selectedTheme = theme)
        return true
    }
}
