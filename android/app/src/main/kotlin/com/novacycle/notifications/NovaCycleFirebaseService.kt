package com.novacycle.notifications

import android.content.Context
import android.util.Log
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

/**
 * Receives signal pushes and persists refreshed tokens for MainActivity to
 * register with the backend. Firebase initialization is project-configured;
 * this class is harmless in builds where that configuration is absent.
 */
class NovaCycleFirebaseService : FirebaseMessagingService() {

    companion object {
        private const val TAG = "NovaCycleFCM"
        const val PREFS_NAME = "novacycle_fcm"
        const val PREF_TOKEN = "fcm_token"
        const val PREF_NEEDS_REGISTRATION = "needs_registration"
    }

    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        val data = remoteMessage.data
        if (data.isEmpty() && remoteMessage.notification == null) return

        val signalType = data["signal_type"] ?: "neutral"
        val gaugeType = data["gauge_type"] ?: "long"
        val confidence = data["confidence"]?.toFloatOrNull() ?: 0.5f
        val isExtended = data["is_extended_hours"]?.lowercase() == "true"
        val sessionType = data["session_type"] ?: "regular"
        val macroOverride = data["macro_override"]?.lowercase() == "true"

        val channel = when {
            macroOverride -> NotificationHelper.CHANNEL_LONG_SIGNALS
            isExtended -> NotificationHelper.CHANNEL_EXTENDED
            gaugeType == "short" -> NotificationHelper.CHANNEL_SHORT_SIGNALS
            else -> NotificationHelper.CHANNEL_LONG_SIGNALS
        }

        NotificationHelper.showSignalNotification(
            context = applicationContext,
            signalType = signalType,
            gaugeType = gaugeType,
            confidence = confidence,
            isExtended = isExtended,
            sessionType = sessionType,
            macroOverrideApplied = macroOverride,
            channelId = channel,
        )
    }

    override fun onNewToken(token: String) {
        Log.d(TAG, "FCM token refreshed")
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_TOKEN, token)
            .putBoolean(PREF_NEEDS_REGISTRATION, true)
            .apply()
    }
}