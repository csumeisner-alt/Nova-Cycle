package com.novacycle.notifications

/**
 * NovaCycleFirebaseService — Firebase Cloud Messaging handler.
 *
 * Currently a stub while google-services.json is pending.
 *
 * To restore full FCM functionality:
 *   1. Add android/app/google-services.json (Firebase Console → project → Android app → com.novacycle)
 *   2. Un-comment alias(libs.plugins.google.services) in app/build.gradle.kts
 *   3. Un-comment the two Firebase dependency lines in app/build.gradle.kts
 *   4. Un-comment the FCM <service> block in AndroidManifest.xml
 *   5. Replace this file with the full implementation below (or un-comment it):
 *
 * ──────────────────────────────────────────────────────────────────────────────
 * import android.content.Context
 * import android.util.Log
 * import com.google.firebase.messaging.FirebaseMessagingService
 * import com.google.firebase.messaging.RemoteMessage
 *
 * class NovaCycleFirebaseService : FirebaseMessagingService() {
 *
 *     companion object {
 *         private const val TAG = "NovaCycleFCM"
 *         const val PREFS_NAME = "novacycle_fcm"
 *         const val PREF_TOKEN = "fcm_token"
 *         const val PREF_NEEDS_REGISTRATION = "needs_registration"
 *     }
 *
 *     override fun onMessageReceived(remoteMessage: RemoteMessage) {
 *         val data = remoteMessage.data
 *         if (data.isEmpty() && remoteMessage.notification == null) return
 *
 *         val signalType = data["signal_type"] ?: "neutral"
 *         val gaugeType  = data["gauge_type"]  ?: "long"
 *         val confidence = data["confidence"]?.toFloatOrNull() ?: 0.5f
 *         val isExtended = data["is_extended_hours"]?.lowercase() == "true"
 *         val sessionType    = data["session_type"]    ?: "regular"
 *         val macroOverride  = data["macro_override"]?.lowercase() == "true"
 *
 *         val channel = when {
 *             macroOverride        -> NotificationHelper.CHANNEL_LONG_SIGNALS
 *             isExtended           -> NotificationHelper.CHANNEL_EXTENDED
 *             gaugeType == "long"  -> NotificationHelper.CHANNEL_LONG_SIGNALS
 *             gaugeType == "short" -> NotificationHelper.CHANNEL_SHORT_SIGNALS
 *             else                 -> NotificationHelper.CHANNEL_LONG_SIGNALS
 *         }
 *
 *         NotificationHelper.showSignalNotification(
 *             context = applicationContext,
 *             signalType = signalType,
 *             gaugeType = gaugeType,
 *             confidence = confidence,
 *             isExtended = isExtended,
 *             sessionType = sessionType,
 *             macroOverrideApplied = macroOverride,
 *             channelId = channel
 *         )
 *     }
 *
 *     override fun onNewToken(token: String) {
 *         applicationContext
 *             .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
 *             .edit()
 *             .putString(PREF_TOKEN, token)
 *             .putBoolean(PREF_NEEDS_REGISTRATION, true)
 *             .apply()
 *     }
 * }
 * ──────────────────────────────────────────────────────────────────────────────
 */

// Companion constants are kept here so MainActivity and NovaCycleApp can reference
// them without change when the full service is restored.
object NovaCycleFirebaseService {
    const val PREFS_NAME = "novacycle_fcm"
    const val PREF_TOKEN = "fcm_token"
    const val PREF_NEEDS_REGISTRATION = "needs_registration"
}
