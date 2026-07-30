package com.novacycle

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import com.novacycle.notifications.NotificationHelper
import com.novacycle.notifications.NovaCycleFirebaseService
import com.google.firebase.messaging.FirebaseMessaging
import dagger.hilt.android.HiltAndroidApp

/**
 * Application class — entry point for Hilt DI graph.
 * Creates the four notification channels used for different signal types.
 *
 * FCM token acquisition is best-effort: a build without project-specific
 * Firebase configuration continues to work but cannot receive pushes.
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

    private fun fetchFcmToken() {
        try {
            FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
                if (!task.isSuccessful) {
                    Log.w(TAG, "FCM token unavailable; configure Firebase before enabling push")
                    return@addOnCompleteListener
                }
                getSharedPreferences(NovaCycleFirebaseService.PREFS_NAME, MODE_PRIVATE)
                    .edit()
                    .putString(NovaCycleFirebaseService.PREF_TOKEN, task.result)
                    .putBoolean(NovaCycleFirebaseService.PREF_NEEDS_REGISTRATION, true)
                    .apply()
                Log.d(TAG, "FCM token acquired; MainActivity will register it")
            }
        } catch (exception: Exception) {
            Log.w(TAG, "Firebase is not configured; push notifications remain disabled")
        }
    }
}
