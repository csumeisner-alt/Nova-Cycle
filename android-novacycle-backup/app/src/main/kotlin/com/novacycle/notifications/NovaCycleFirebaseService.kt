package com.novacycle.notifications

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
 * - is_extended: "true" | "false"
 * - session_type: "regular" | "pre_market" | "after_hours"
 * - score: int string
 * - macro_override: "true" | "false"
 */
class NovaCycleFirebaseService : FirebaseMessagingService() {

    companion object {
        private const val TAG = "NovaCycleFCM"
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
        val isExtended = data["is_extended"]?.lowercase() == "true"
        val sessionType = data["session_type"] ?: "regular"
        val macroOverride = data["macro_override"]?.lowercase() == "true"

        Log.d(TAG, "Signal: $signalType | Gauge: $gaugeType | Confidence: $confidence | Extended: $isExtended")

        // Determine which notification channel to use
        val channel = when {
            macroOverride   -> NotificationHelper.CHANNEL_LONG_SIGNALS
            isExtended      -> NotificationHelper.CHANNEL_EXTENDED
            gaugeType == "long"  -> NotificationHelper.CHANNEL_LONG_SIGNALS
            gaugeType == "short" -> NotificationHelper.CHANNEL_SHORT_SIGNALS
            else            -> NotificationHelper.CHANNEL_LONG_SIGNALS
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
        // TODO: Send new token to NovaCycle backend via API when authentication is added.
        // For now, the token is available in the app for manual configuration.
        // Example:
        //   CoroutineScope(Dispatchers.IO).launch {
        //       repository.updateFcmToken(token)
        //   }
    }
}
