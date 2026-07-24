package com.novacycle

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.notifications.NovaCycleFirebaseService
import com.novacycle.ui.navigation.NovaCycleNavHost
import com.novacycle.ui.theme.NovaCycleTheme
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
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
            val result = repository.registerDeviceToken(token, deviceName)
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
}
