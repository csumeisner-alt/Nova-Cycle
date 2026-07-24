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
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean
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

    /**
     * Guards against two concurrent registration coroutines running simultaneously.
     * Set to true when a registration coroutine is launched; reset to false in the
     * finally block when it completes (success or failure). A second call to
     * registerFcmTokenIfNeeded() while a coroutine is already in flight is a no-op.
     *
     * INVARIANT — this field MUST remain a plain in-memory AtomicBoolean.
     * Do NOT persist it to SharedPreferences, DataStore, or any other on-disk store.
     *
     * Reason: if the process is killed mid-coroutine (OOM, force-stop, ANR), the
     * JVM heap is discarded and this flag resets to false on the next launch —
     * which is the correct, safe behaviour. If it were ever persisted to disk and
     * the process were killed between the write and the finally-block reset, the
     * flag would stay true across reboots and silently block every future FCM
     * registration attempt, preventing the device from ever receiving push
     * notifications again.
     */
    private val registrationInFlight = AtomicBoolean(false)

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

        if (token == null) {
            Log.d(TAG, "FCM token not yet available (Firebase disabled or not yet configured)")
            return
        }

        // Prevent a second coroutine from launching while one is already running.
        // compareAndSet returns false if the flag was already true, meaning registration
        // is in flight — in that case this call is a no-op.
        if (!registrationInFlight.compareAndSet(false, true)) {
            Log.d(TAG, "FCM token registration already in flight — skipping duplicate call")
            return
        }

        val needsRegistration = prefs.getBoolean(NovaCycleFirebaseService.PREF_NEEDS_REGISTRATION, false)
        val deviceName = android.os.Build.MODEL

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                if (needsRegistration) {
                    // Token is new or was explicitly marked for re-registration — register immediately.
                    registerToken(token, deviceName, prefs)
                } else {
                    // Token was previously registered. Verify it is still known to the backend
                    // (guards against a backend DB reset wiping the device_tokens table).
                    val checkResult = repository.checkDeviceToken(token)
                    checkResult
                        .onSuccess { found ->
                            if (found) {
                                Log.d(TAG, "FCM token confirmed present on backend")
                            } else {
                                // Backend returned 404 — DB was reset; re-register now.
                                Log.w(TAG, "FCM token missing from backend (DB may have been reset) — re-registering")
                                registerToken(token, deviceName, prefs)
                            }
                        }
                        .onFailure { e ->
                            // Backend unreachable — leave needsRegistration flag as-is and retry next launch.
                            Log.e(TAG, "FCM token check failed (backend unreachable, will retry on next launch): ${e.message}")
                        }
                }
            } finally {
                // Always release the lock so the next legitimate call (e.g. after a token
                // refresh) can proceed.
                registrationInFlight.set(false)
            }
        }
    }

    /**
     * Send the FCM token to the backend and, on success, clear the pending-registration flag.
     */
    private suspend fun registerToken(
        token: String,
        deviceName: String,
        prefs: android.content.SharedPreferences,
    ) {
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
                    apiBaseUrl = prefs[SettingsViewModel.KEY_API_BASE_URL] ?: com.novacycle.BuildConfig.API_BASE_URL
                )
            }
            .first()
}
