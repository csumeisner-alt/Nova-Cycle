package com.novacycle.domain.model

/**
 * User-configurable sensitivity and display settings.
 * Persisted via DataStore. Applied client-side in ViewModels and use cases.
 */
data class SensitivitySettings(
    /** Minimum BUY confidence to display/notify (50–80) */
    val buyThreshold: Int = 70,
    /** Maximum SELL confidence to display/notify (-80 to -50, stored as negative) */
    val sellThreshold: Int = -70,
    /** Whether to include extended-hours signals in charts/notifications */
    val extendedHoursEnabled: Boolean = true,
    /** How to weight indicators vs ML in score interpretation */
    val weightingMode: WeightingMode = WeightingMode.BALANCED,
    /** Chart smoothing algorithm applied to confidence history */
    val smoothingMode: SmoothingMode = SmoothingMode.RAW,
    /** How much detail to show in signal story cards */
    val storyCardLevel: StoryLevel = StoryLevel.SIMPLE,
    /** How sensitive push notifications are */
    val notificationSensitivity: NotifSensitivity = NotifSensitivity.STANDARD,
    /** Whether to send notifications for extended-hours signals */
    val extendedHoursNotifications: Boolean = true,
    /** Backend API base URL — overridable from settings screen */
    val apiBaseUrl: String = "http://10.0.2.2:8080/api/"
)

enum class WeightingMode {
    BALANCED,
    INDICATOR_HEAVY,
    ML_HEAVY
}

enum class SmoothingMode {
    RAW,
    LIGHT,
    EMA,
    HEAVY
}

enum class StoryLevel {
    SIMPLE,
    ADVANCED,
    EXPERT
}

enum class NotifSensitivity {
    STANDARD,
    HIGH,
    LOW
}
