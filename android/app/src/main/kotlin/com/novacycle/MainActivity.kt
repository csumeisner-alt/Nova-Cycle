package com.novacycle

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.WeightingMode
import com.novacycle.domain.model.SmoothingMode
import com.novacycle.domain.model.StoryLevel
import com.novacycle.domain.model.NotifSensitivity
import com.novacycle.notifications.NovaCycleFirebaseService
import com.novacycle.ui.navigation.NovaCycleNavHost
import com.novacycle.ui.theme.NovaCycleTheme
import com.novacycle.viewmodel.SettingsViewModel
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Single-activity architecture. All navigation is handled by Jetpack Navigation Compose.
 * Hilt injects dependencies throughout the Compose tree via hiltViewModel().
 *
 * FCM token registration is a no-op while Firebase is disabled (google-services.json absent).
 * When Firebase is re-enabled, registerFcmTokenIfNeeded() will automatically start sending
 * the device token to the backend on each launch.
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    @Inject
    lateinit var repository: NovaCycleRepository

    @Inject
    lateinit var dataStore: DataStore<Preferences>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Register FCM token with the backend (no-op when Firebase is disabled)
        registerFcmTokenIfNeeded()

        setContent {
            NovaCycleTheme {
                NovaCycleNavHost()
            }
        }
    }

    /**
     * Called when the activity is already running and a notification tap brings it to the
     * foreground. Re-registers the token in case it was refreshed while backgrounded.
     */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        registerFcmTokenIfNeeded()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Private
    // ──────────────────────────────────────────────────────────────────────────

    private fun registerFcmTokenIfNeeded() {
        val prefs = getSharedPreferences(NovaCycleFirebaseService.PREFS_NAME, MODE_PRIVATE)
        val token = prefs.getString(NovaCycleFirebaseService.PREF_TOKEN, null)
        val needsRegistration = prefs.getBoolean(NovaCycleFirebaseService.PREF_NEEDS_REGISTRATION, false)

        if (token == null || !needsRegistration) {
            Log.d(TAG, "FCM token not available or already registered (Firebase disabled or not yet configured)")
            return
        }

        val deviceName = android.os.Build.MODEL

        CoroutineScope(Dispatchers.IO).launch {
            // Read current sensitivity settings so the backend receives up-to-date
            // notification preferences alongside the device token.
            val settings = readCurrentSettings()

            val result = repository.registerDeviceToken(token, deviceName, settings)
            result.onSuccess {
                Log.d(TAG, "FCM token registered with backend successfully")
                prefs.edit()
                    .putBoolean(NovaCycleFirebaseService.PREF_NEEDS_REGISTRATION, false)
                    .apply()
            }
            result.onFailure { e ->
                Log.e(TAG, "FCM token registration failed (will retry on next launch): ${e.message}")
            }
        }
    }

    /**
     * Read a snapshot of the user's sensitivity settings from DataStore.
     * Used to include notification preferences when registering the FCM token.
     */
    private suspend fun readCurrentSettings(): SensitivitySettings =
        dataStore.data
            .catch { emit(androidx.datastore.preferences.core.emptyPreferences()) }
            .map { prefs ->
                SensitivitySettings(
                    buyThreshold = prefs[SettingsViewModel.KEY_BUY_THRESHOLD] ?: 70,
                    sellThreshold = prefs[SettingsViewModel.KEY_SELL_THRESHOLD] ?: -70,
                    extendedHoursEnabled = prefs[SettingsViewModel.KEY_EXTENDED_HOURS] ?: true,
                    weightingMode = WeightingMode.valueOf(
                        prefs[SettingsViewModel.KEY_WEIGHTING_MODE] ?: WeightingMode.BALANCED.name
                    ),
                    smoothingMode = SmoothingMode.valueOf(
                        prefs[SettingsViewModel.KEY_SMOOTHING_MODE] ?: SmoothingMode.RAW.name
                    ),
                    storyCardLevel = StoryLevel.valueOf(
                        prefs[SettingsViewModel.KEY_STORY_LEVEL] ?: StoryLevel.SIMPLE.name
                    ),
                    notificationSensitivity = NotifSensitivity.valueOf(
                        prefs[SettingsViewModel.KEY_NOTIF_SENSITIVITY] ?: NotifSensitivity.STANDARD.name
                    ),
                    extendedHoursNotifications = prefs[SettingsViewModel.KEY_EXTENDED_NOTIF] ?: true,
                    apiBaseUrl = prefs[SettingsViewModel.KEY_API_BASE_URL] ?: "http://10.0.2.2:8080/api/"
                )
            }
            .first()
}
