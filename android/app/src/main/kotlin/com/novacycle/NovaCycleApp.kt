package com.novacycle

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import com.novacycle.notifications.NotificationHelper
import dagger.hilt.android.HiltAndroidApp

/**
 * Application class — entry point for Hilt DI graph.
 * Creates the four notification channels used for different signal types.
 *
 * Firebase / FCM is temporarily disabled while google-services.json is pending.
 * To re-enable:
 *   1. Add android/app/google-services.json (from Firebase Console, package com.novacycle)
 *   2. Un-comment alias(libs.plugins.google.services) in app/build.gradle.kts
 *   3. Un-comment the two Firebase dependency lines in app/build.gradle.kts
 *   4. Un-comment the FCM service block in AndroidManifest.xml
 *   5. Restore the Firebase imports + fetchFcmToken() call in this file
 *   6. Restore NovaCycleFirebaseService.kt from the full version
 */
@HiltAndroidApp
class NovaCycleApp : Application() {

    companion object {
        private const val TAG = "NovaCycleApp"
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
        Log.d(TAG, "NovaCycle started (Firebase disabled — add google-services.json to enable push notifications)")
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
}
