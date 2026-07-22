package com.novacycle

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import com.novacycle.notifications.NotificationHelper
import dagger.hilt.android.HiltAndroidApp

/**
 * Application class — entry point for Hilt DI graph.
 * Creates the four notification channels used for different signal types.
 * Channel IDs are sourced from NotificationHelper to keep them in sync.
 */
@HiltAndroidApp
class NovaCycleApp : Application() {

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
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
