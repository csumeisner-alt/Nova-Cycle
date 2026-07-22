package com.novacycle.viewmodel

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.domain.model.*
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Manages user sensitivity settings persisted in DataStore.
 * All reads are hot flows; writes are fire-and-forget coroutines.
 */
@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val dataStore: DataStore<androidx.datastore.preferences.core.Preferences>
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

    fun updateBuyThreshold(value: Int) = save { prefs ->
        prefs[KEY_BUY_THRESHOLD] = value.coerceIn(50, 80)
    }

    fun updateSellThreshold(value: Int) = save { prefs ->
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

    fun updateNotifSensitivity(sensitivity: NotifSensitivity) = save { prefs ->
        prefs[KEY_NOTIF_SENSITIVITY] = sensitivity.name
    }

    fun updateExtendedHoursNotifications(enabled: Boolean) = save { prefs ->
        prefs[KEY_EXTENDED_NOTIF] = enabled
    }

    fun updateApiBaseUrl(url: String) = save { prefs ->
        prefs[KEY_API_BASE_URL] = url
    }

    fun resetToDefaults() = save { prefs ->
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

    private fun save(block: (MutablePreferences) -> Unit) {
        viewModelScope.launch {
            dataStore.edit { block(it) }
        }
    }
}
