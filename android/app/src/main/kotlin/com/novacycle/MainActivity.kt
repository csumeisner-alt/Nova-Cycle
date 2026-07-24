package com.novacycle

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.novacycle.data.remote.models.RegisterDeviceRequest
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
 * On startup, reads the FCM token stored by NovaCycleFirebaseService/NovaCycleApp and
 * registers it with the NovaCycle backend so BUY/SELL push notifications are delivered.
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

        // Register FCM token with the backend (idempotent — safe on every launch)
        registerFcmTokenIfNeeded()

        setContent {
            NovaCycleTheme {
                NovaCycleNavHost()
            }
        }
    }

    /**
     * Called when the activity is already running and a notification tap brings it to the
     * foreground. We re-register the token in case it was refreshed while backgrounded.
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
            Log.d(TAG, "FCM token already registered or not yet available")
            return
        }

        val deviceName = android.os.Build.MODEL

        CoroutineScope(Dispatchers.IO).launch {
            // repository.registerDeviceToken returns Result<Unit> — inspect it directly.
            // Do NOT wrap in runCatching here; that would always succeed and mask
            // backend failures, causing needs_registration to be cleared incorrectly.
            val result = repository.registerDeviceToken(token, deviceName)
            result.onSuccess {
                Log.d(TAG, "FCM token registered with backend successfully")
                // Only clear the flag on confirmed backend success
                prefs.edit()
                    .putBoolean(NovaCycleFirebaseService.PREF_NEEDS_REGISTRATION, false)
                    .apply()
            }
            result.onFailure { e ->
                Log.e(TAG, "FCM token registration failed (will retry on next launch): ${e.message}")
                // Leave needs_registration=true so MainActivity retries on the next launch
            }
        }
    }
}
