package com.novacycle

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import com.google.firebase.messaging.FirebaseMessaging
import com.novacycle.notifications.NotificationHelper
import com.novacycle.notifications.NovaCycleFirebaseService
import dagger.hilt.android.HiltAndroidApp

/**
 * Application class — entry point for Hilt DI graph.
 * Creates the four notification channels used for different signal types.
 * Also triggers an FCM token fetch so onNewToken fires if the token is new or
 * needs refreshing, storing it for registration on the next MainActivity launch.
 */
@HiltAndroidApp
class NovaCycleApp : Application() {

    companion object {
        private const val TAG = "NovaCycleApp"
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
        fetchFcmToken()
    }

    private fun createNotificationChannels() {
        val manager = getSystemService(NotificationManager::class.java) ?: return

        val channels = listOf(
            NotificationChannel(
                NotificationHelper.CHANNEL_LONG_SIGNALS,
                "Long-Term Signals",
                NotificationManager.IMPORTANCE_HIGH
            ).apply { description = "BUY/SELL signals from the long-trend gauge" },

            NotificationChannel(
                NotificationHelper.CHANNEL_SHORT_SIGNALS,
                "Short-Term Signals",
                NotificationManager.IMPORTANCE_HIGH
            ).apply { description = "BUY/SELL signals from the short-trend gauge" },

            NotificationChannel(
                NotificationHelper.CHANNEL_EXTENDED,
                "Extended-Hours Signals",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply { description = "Signals triggered during pre/post-market extended hours" },

            NotificationChannel(
                NotificationHelper.CHANNEL_MOMENTUM,
                "Confidence Momentum",
                NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Alerts when confidence momentum shifts significantly" }
        )

        channels.forEach { manager.createNotificationChannel(it) }
    }

    /**
     * Ask Firebase for the current registration token.
     * If the token is fresh (app just installed, or token rotated), Firebase
     * calls NovaCycleFirebaseService.onNewToken which saves it to SharedPreferences.
     * MainActivity then reads SharedPreferences and registers the token with our backend.
     */
    private fun fetchFcmToken() {
        FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
            if (!task.isSuccessful) {
                Log.w(TAG, "FCM token fetch failed: ${task.exception?.message}")
                return@addOnCompleteListener
            }
            val token = task.result ?: return@addOnCompleteListener
            Log.d(TAG, "FCM token available: ${token.take(20)}...")

            // Store the token — MainActivity will register it with the backend
            getSharedPreferences(NovaCycleFirebaseService.PREFS_NAME, MODE_PRIVATE)
                .edit()
                .putString(NovaCycleFirebaseService.PREF_TOKEN, token)
                .putBoolean(NovaCycleFirebaseService.PREF_NEEDS_REGISTRATION, true)
                .apply()
        }
    }
}
