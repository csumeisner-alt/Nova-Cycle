package com.novacycle.viewmodel

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.model.*
import com.novacycle.notifications.NovaCycleFirebaseService
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Manages user sensitivity settings persisted in DataStore.
 * All reads are hot flows; writes are fire-and-forget coroutines.
 *
 * When notification-relevant preferences change (sensitivity level, extended-hours
 * toggle), the updated preferences are immediately re-synced to the backend via
 * [NovaCycleRepository.registerDeviceToken] so the backend can apply them to the
 * next push notification without waiting for an app restart.
 */
@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val dataStore: DataStore<Preferences>,
    private val repository: NovaCycleRepository,
    @ApplicationContext private val appContext: Context
) : ViewModel() {

    // DataStore preference keys
    companion object {
        val KEY_BUY_THRESHOLD = intPreferencesKey("buy_threshold")
        val KEY_SELL_THRESHOLD = intPreferencesKey("sell_threshold")
        val KEY_EXTENDED_HOURS = booleanPreferencesKey("extended_hours_enabled")
        val KEY_WEIGHTING_MODE = stringPreferencesKey("weighting_mode")
        val KEY_SMOOTHING_MODE = stringPreferencesKey("smoothing_mode")
        val KEY_STORY_LEVEL = stringPreferencesKey("story_level")
        val KEY_NOTIF_SENSITIVITY = stringPreferencesKey("notif_sensitivity")
        val KEY_EXTENDED_NOTIF = booleanPreferencesKey("extended_hours_notifications")
        val KEY_API_BASE_URL = stringPreferencesKey("api_base_url")
    }

    val settings: StateFlow<SensitivitySettings> = dataStore.data
        .catch { emit(emptyPreferences()) }
        .map { prefs ->
            SensitivitySettings(
                buyThreshold = prefs[KEY_BUY_THRESHOLD] ?: 70,
                sellThreshold = prefs[KEY_SELL_THRESHOLD] ?: -70,
                extendedHoursEnabled = prefs[KEY_EXTENDED_HOURS] ?: true,
                weightingMode = WeightingMode.valueOf(
                    prefs[KEY_WEIGHTING_MODE] ?: WeightingMode.BALANCED.name
                ),
                smoothingMode = SmoothingMode.valueOf(
                    prefs[KEY_SMOOTHING_MODE] ?: SmoothingMode.RAW.name
                ),
                storyCardLevel = StoryLevel.valueOf(
                    prefs[KEY_STORY_LEVEL] ?: StoryLevel.SIMPLE.name
                ),
                notificationSensitivity = NotifSensitivity.valueOf(
                    prefs[KEY_NOTIF_SENSITIVITY] ?: NotifSensitivity.STANDARD.name
                ),
                extendedHoursNotifications = prefs[KEY_EXTENDED_NOTIF] ?: true,
                apiBaseUrl = prefs[KEY_API_BASE_URL] ?: "http://10.0.2.2:8080/api/"
            )
        }
        .stateIn(viewModelScope, SharingStarted.Eagerly, SensitivitySettings())

    fun updateBuyThreshold(value: Int) = saveAndSync { prefs ->
        prefs[KEY_BUY_THRESHOLD] = value.coerceIn(50, 80)
    }

    fun updateSellThreshold(value: Int) = saveAndSync { prefs ->
        // Store as negative integer
        prefs[KEY_SELL_THRESHOLD] = -value.coerceIn(50, 80)
    }

    fun updateExtendedHoursEnabled(enabled: Boolean) = save { prefs ->
        prefs[KEY_EXTENDED_HOURS] = enabled
    }

    fun updateWeightingMode(mode: WeightingMode) = save { prefs ->
        prefs[KEY_WEIGHTING_MODE] = mode.name
    }

    fun updateSmoothingMode(mode: SmoothingMode) = save { prefs ->
        prefs[KEY_SMOOTHING_MODE] = mode.name
    }

    fun updateStoryLevel(level: StoryLevel) = save { prefs ->
        prefs[KEY_STORY_LEVEL] = level.name
    }

    fun updateNotifSensitivity(sensitivity: NotifSensitivity) = saveAndSync { prefs ->
        prefs[KEY_NOTIF_SENSITIVITY] = sensitivity.name
    }

    fun updateExtendedHoursNotifications(enabled: Boolean) = saveAndSync { prefs ->
        prefs[KEY_EXTENDED_NOTIF] = enabled
    }

    fun updateApiBaseUrl(url: String) = save { prefs ->
        prefs[KEY_API_BASE_URL] = url
    }

    fun resetToDefaults() = saveAndSync { prefs ->
        val defaults = SensitivitySettings()
        prefs[KEY_BUY_THRESHOLD] = defaults.buyThreshold
        prefs[KEY_SELL_THRESHOLD] = defaults.sellThreshold
        prefs[KEY_EXTENDED_HOURS] = defaults.extendedHoursEnabled
        prefs[KEY_WEIGHTING_MODE] = defaults.weightingMode.name
        prefs[KEY_SMOOTHING_MODE] = defaults.smoothingMode.name
        prefs[KEY_STORY_LEVEL] = defaults.storyCardLevel.name
        prefs[KEY_NOTIF_SENSITIVITY] = defaults.notificationSensitivity.name
        prefs[KEY_EXTENDED_NOTIF] = defaults.extendedHoursNotifications
        prefs[KEY_API_BASE_URL] = defaults.apiBaseUrl
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Internals
    // ──────────────────────────────────────────────────────────────────────────

    /** Persist to DataStore only. */
    private fun save(block: (MutablePreferences) -> Unit) {
        viewModelScope.launch {
            dataStore.edit { block(it) }
        }
    }

    /**
     * Persist to DataStore, then re-sync notification preferences with the backend.
     * Used for settings that affect which signals the backend sends to this device.
     * No-op (beyond the save) when no FCM token is registered yet.
     */
    private fun saveAndSync(block: (MutablePreferences) -> Unit) {
        viewModelScope.launch {
            dataStore.edit { block(it) }
            // Re-read the updated settings snapshot and push to the backend.
            syncPreferencesWithBackend()
        }
    }

    /**
     * Re-register the FCM token with the backend carrying the current notification
     * preferences. This updates the backend's per-device thresholds immediately so
     * the next signal respects the new settings without waiting for an app restart.
     *
     * Silently skips when no FCM token is available (Firebase not yet configured).
     */
    private suspend fun syncPreferencesWithBackend() {
        val sharedPrefs = appContext.getSharedPreferences(
            NovaCycleFirebaseService.PREFS_NAME,
            Context.MODE_PRIVATE
        )
        val token = sharedPrefs.getString(NovaCycleFirebaseService.PREF_TOKEN, null)
            ?: return  // Firebase not yet configured — nothing to sync

        val currentSettings = settings.value
        val deviceName = android.os.Build.MODEL

        repository.registerDeviceToken(token, deviceName, currentSettings)
            .onFailure { e ->
                // Non-fatal — preferences will be re-synced on next launch
                android.util.Log.w(
                    "SettingsViewModel",
                    "Failed to sync notification preferences with backend: ${e.message}"
                )
            }
    }
}
