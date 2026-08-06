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
        // Install the crash reporter BEFORE anything else so exceptions during
        // startup (Hilt graph construction, DataStore init, etc.) are captured too.
        installCrashReporter()
        super.onCreate()
        createNotificationChannels()
        fetchFcmToken()
    }

    /**
     * Installs a global uncaught-exception handler that writes the full stack
     * trace to [filesDir]/crash_log.txt before delegating to the system handler.
     *
     * To read the file from a debug build:
     *   adb shell run-as com.novacycle cat /data/data/com.novacycle/files/crash_log.txt
     *
     * The file is also printed at ERROR level so logcat captures it without ADB.
     * This shim is intentionally kept permanent: a production crash log costs
     * almost nothing but is invaluable when a release APK crashes silently.
     */
    private fun installCrashReporter() {
        val systemHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                val report = buildString {
                    appendLine("=== NovaCycle Crash Log ===")
                    appendLine("Thread : ${thread.name}")
                    appendLine("Time   : ${java.util.Date()}")
                    appendLine("Build  : ${android.os.Build.MODEL} / Android ${android.os.Build.VERSION.SDK_INT}")
                    appendLine()
                    appendLine(throwable.stackTraceToString())
                }
                val logFile = java.io.File(filesDir, "crash_log.txt")
                logFile.writeText(report)
                // Also dump to logcat — visible via `adb logcat -s NovaCycleCrash`
                Log.e(TAG, "CRASH CAPTURED → ${logFile.absolutePath}\n$report")
            } catch (ignore: Exception) {
                // Never let the reporter itself prevent the system handler from running.
            }
            systemHandler?.uncaughtException(thread, throwable)
        }
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
