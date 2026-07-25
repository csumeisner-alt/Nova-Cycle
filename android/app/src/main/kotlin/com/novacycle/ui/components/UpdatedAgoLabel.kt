package com.novacycle.ui.components

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

/**
 * Small relative "Updated X ago" label used on every data screen.
 * Renders nothing until the screen has completed at least one successful fetch.
 * Ticks while visible via [rememberTickingNow].
 */
@Composable
fun UpdatedAgoLabel(
    lastUpdatedAtMillis: Long?,
    modifier: Modifier = Modifier
) {
    if (lastUpdatedAtMillis == null) return
    val now = rememberTickingNow()
    Text(
        text = "Updated ${formatRelativeAge(now, lastUpdatedAtMillis)}",
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
        modifier = modifier
    )
}
