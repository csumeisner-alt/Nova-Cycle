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
