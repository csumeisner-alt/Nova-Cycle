package com.novacycle.ui.components

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import kotlinx.coroutines.delay

/**
 * Current time (epoch millis) that re-emits every [intervalMillis] so
 * relative "last updated X ago" labels keep ticking while on screen.
 */
@Composable
fun rememberTickingNow(intervalMillis: Long = 30_000L): Long {
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(intervalMillis) {
        while (true) {
            delay(intervalMillis)
            now = System.currentTimeMillis()
        }
    }
    return now
}

/** How stale a piece of data is, relative to configurable thresholds. */
enum class StalenessLevel { FRESH, WARNING, CRITICAL }

/** Default thresholds: amber past 5 min, red past 15 min. */
const val DEFAULT_WARNING_THRESHOLD_MILLIS: Long = 5 * 60_000L
const val DEFAULT_CRITICAL_THRESHOLD_MILLIS: Long = 15 * 60_000L

/**
 * Classifies the age of [thenMillis] relative to [nowMillis] against
 * warning/critical thresholds. Ages at or past a threshold trigger it.
 */
fun stalenessLevel(
    nowMillis: Long,
    thenMillis: Long,
    warningThresholdMillis: Long = DEFAULT_WARNING_THRESHOLD_MILLIS,
    criticalThresholdMillis: Long = DEFAULT_CRITICAL_THRESHOLD_MILLIS
): StalenessLevel {
    val age = (nowMillis - thenMillis).coerceAtLeast(0L)
    return when {
        age >= criticalThresholdMillis -> StalenessLevel.CRITICAL
        age >= warningThresholdMillis -> StalenessLevel.WARNING
        else -> StalenessLevel.FRESH
    }
}

/**
 * Human-readable age of [thenMillis] relative to [nowMillis],
 * e.g. "just now", "12 min ago", "3 h 5 min ago", "2 days ago".
 */
fun formatRelativeAge(nowMillis: Long, thenMillis: Long): String {
    val diffSec = ((nowMillis - thenMillis) / 1000L).coerceAtLeast(0L)
    val minutes = diffSec / 60
    val hours = minutes / 60
    val days = hours / 24
    return when {
        diffSec < 60 -> "just now"
        minutes < 60 -> "$minutes min ago"
        hours < 24 -> {
            val remMin = minutes % 60
            if (remMin == 0L) "$hours h ago" else "$hours h $remMin min ago"
        }
        else -> if (days == 1L) "1 day ago" else "$days days ago"
    }
}
