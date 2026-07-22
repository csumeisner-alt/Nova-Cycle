package com.novacycle.notifications

import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat
import com.novacycle.MainActivity

/**
 * NotificationHelper — builds and shows NovaCycle signal notifications.
 *
 * Channel IDs:
 * - CHANNEL_LONG_SIGNALS:  Long-term BUY/SELL alerts
 * - CHANNEL_SHORT_SIGNALS: Short-term BUY/SELL alerts
 * - CHANNEL_EXTENDED:      Extended-hours signal alerts
 * - CHANNEL_MOMENTUM:      Confidence momentum change alerts
 */
object NotificationHelper {

    const val CHANNEL_LONG_SIGNALS  = "novacycle_long_signals"
    const val CHANNEL_SHORT_SIGNALS = "novacycle_short_signals"
    const val CHANNEL_EXTENDED      = "novacycle_extended"
    const val CHANNEL_MOMENTUM      = "novacycle_momentum"

    private var notifId = 1000

    /**
     * Show a BUY or SELL signal notification.
     *
     * @param signalType "buy" or "sell"
     * @param gaugeType  "long" or "short"
     * @param confidence 0.0–1.0
     * @param isExtended True if signal came from extended-hours session
     * @param sessionType "regular" | "pre_market" | "after_hours"
     * @param macroOverrideApplied True if macro safety layer was triggered
     * @param channelId  Which notification channel to post to
     */
    fun showSignalNotification(
        context: Context,
        signalType: String,
        gaugeType: String,
        confidence: Float,
        isExtended: Boolean,
        sessionType: String = "regular",
        macroOverrideApplied: Boolean = false,
        channelId: String = CHANNEL_LONG_SIGNALS
    ) {
        val notifManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val confidencePct = (confidence * 100).toInt()
        val gaugeLabel = if (gaugeType == "long") "Long-Trend" else "Short-Trend"
        val signalLabel = signalType.uppercase()

        // Title: "NovaCycle: LONG BUY" or "NovaCycle: SHORT SELL"
        val title = "NovaCycle: $gaugeLabel $signalLabel"

        // Body with confidence + session context
        val sessionNote = when {
            macroOverrideApplied    -> " ⚠ Macro override active"
            isExtended              -> " · ${sessionType.replace("_", "-")} session"
            else                    -> ""
        }
        val body = "Confidence: $confidencePct%$sessionNote"

        // Color: green for BUY, red for SELL
        val colorInt = if (signalType == "buy") 0xFF00C853.toInt() else 0xFFD50000.toInt()

        // Tap intent: opens MainActivity (DualGaugeScreen)
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra("navigate_to", "dual_gauge")
        }
        val pendingIntent = PendingIntent.getActivity(
            context,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        // Big text style for extended info
        val bigTextStyle = NotificationCompat.BigTextStyle()
            .bigText(
                "$body\n" +
                "Gauge: $gaugeLabel | Score confidence: $confidencePct%\n" +
                (if (isExtended) "⚡ Extended-hours signal — lower liquidity weight applied\n" else "") +
                (if (macroOverrideApplied) "⚠ Macro override: long-trend is suppressing short-term signal\n" else "") +
                "Tap to view full signal details"
            )
            .setBigContentTitle(title)

        val notification = NotificationCompat.Builder(context, channelId)
            .setSmallIcon(android.R.drawable.ic_dialog_info)  // Replace with custom icon
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(bigTextStyle)
            .setColor(colorInt)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .setPriority(
                if (gaugeType == "long") NotificationCompat.PRIORITY_HIGH
                else NotificationCompat.PRIORITY_DEFAULT
            )
            .build()

        notifManager.notify(notifId++, notification)
    }

    /**
     * Show a confidence momentum alert when confidence spikes or drops significantly.
     *
     * @param ticker     The ticker symbol (currently always "VOO")
     * @param gaugeType  "long" or "short"
     * @param momentum   Confidence change: positive = rising, negative = falling
     * @param currentConf Current confidence value 0.0–1.0
     */
    fun showMomentumAlert(
        context: Context,
        ticker: String,
        gaugeType: String,
        momentum: Float,
        currentConf: Float
    ) {
        val notifManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val direction = if (momentum > 0) "↑ Rising" else "↓ Falling"
        val gaugeLabel = if (gaugeType == "long") "Long-Trend" else "Short-Trend"
        val title = "NovaCycle: $ticker $gaugeLabel Momentum $direction"
        val body = "Confidence: ${(currentConf * 100).toInt()}% (Δ${String.format("%+.1f", momentum * 100)}%)"

        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            context, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(context, CHANNEL_MOMENTUM)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle(title)
            .setContentText(body)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

        notifManager.notify(notifId++, notification)
    }
}
