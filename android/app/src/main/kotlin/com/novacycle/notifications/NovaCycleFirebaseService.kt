package com.novacycle.notifications

import android.content.Context
import android.util.Log
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

/**
 * NovaCycleFirebaseService — handles incoming FCM push notifications.
 *
 * Notification data payload fields (sent by backend):
 * - signal_type: "buy" | "sell"
 * - gauge_type: "long" | "short"
 * - confidence: float string (e.g. "0.84")
 * - is_extended_hours: "true" | "false"
 * - score: float string
 * - gap_type: "gap_up" | "gap_down" | "none"
 * - ticker: "VOO"
 */
class NovaCycleFirebaseService : FirebaseMessagingService() {

    companion object {
        private const val TAG = "NovaCycleFCM"
        const val PREFS_NAME = "novacycle_fcm"
        const val PREF_TOKEN = "fcm_token"
        const val PREF_NEEDS_REGISTRATION = "needs_registration"
    }

    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        Log.d(TAG, "FCM message received from: ${remoteMessage.from}")

        // Extract data payload
        val data = remoteMessage.data
        if (data.isEmpty() && remoteMessage.notification == null) {
            Log.w(TAG, "Empty FCM message received — ignoring")
            return
        }

        val signalType = data["signal_type"] ?: remoteMessage.notification?.title?.lowercase()?.let {
            if (it.contains("buy")) "buy" else if (it.contains("sell")) "sell" else "neutral"
        } ?: "neutral"

        val gaugeType = data["gauge_type"] ?: "long"
        val confidenceStr = data["confidence"] ?: "0.5"
        val confidence = confidenceStr.toFloatOrNull() ?: 0.5f
        val isExtended = data["is_extended_hours"]?.lowercase() == "true"
        val sessionType = data["session_type"] ?: "regular"
        val macroOverride = data["macro_override"]?.lowercase() == "true"

        Log.d(TAG, "Signal: $signalType | Gauge: $gaugeType | Confidence: $confidence | Extended: $isExtended")

        // Determine which notification channel to use
        val channel = when {
            macroOverride        -> NotificationHelper.CHANNEL_LONG_SIGNALS
            isExtended           -> NotificationHelper.CHANNEL_EXTENDED
            gaugeType == "long"  -> NotificationHelper.CHANNEL_LONG_SIGNALS
            gaugeType == "short" -> NotificationHelper.CHANNEL_SHORT_SIGNALS
            else                 -> NotificationHelper.CHANNEL_LONG_SIGNALS
        }

        // Show the notification
        NotificationHelper.showSignalNotification(
            context = applicationContext,
            signalType = signalType,
            gaugeType = gaugeType,
            confidence = confidence,
            isExtended = isExtended,
            sessionType = sessionType,
            macroOverrideApplied = macroOverride,
            channelId = channel
        )
    }

    override fun onNewToken(token: String) {
        Log.d(TAG, "FCM token refreshed: ${token.take(20)}...")

        // Persist the new token and flag it for backend registration.
        // The actual HTTP call is made from MainActivity on the next launch,
        // where Hilt-injected dependencies are available.
        applicationContext
            .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_TOKEN, token)
            .putBoolean(PREF_NEEDS_REGISTRATION, true)
            .apply()
    }
}
