package com.novacycle.ui.components

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight

/** Amber used once data crosses the warning threshold. */
private val WarningColor = Color(0xFFB26A00)

/**
 * Small relative "Updated X ago" label used on every data screen.
 * Renders nothing until the screen has completed at least one successful fetch.
 * Ticks while visible via [rememberTickingNow].
 *
 * The label turns amber once the age crosses [warningThresholdMillis] and red
 * (theme error color) once it crosses [criticalThresholdMillis], so dangerously
 * stale data is impossible to miss.
 */
@Composable
fun UpdatedAgoLabel(
    lastUpdatedAtMillis: Long?,
    modifier: Modifier = Modifier,
    warningThresholdMillis: Long = DEFAULT_WARNING_THRESHOLD_MILLIS,
    criticalThresholdMillis: Long = DEFAULT_CRITICAL_THRESHOLD_MILLIS
) {
    if (lastUpdatedAtMillis == null) return
    val now = rememberTickingNow()
    val level = stalenessLevel(now, lastUpdatedAtMillis, warningThresholdMillis, criticalThresholdMillis)
    val color = when (level) {
        StalenessLevel.FRESH -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        StalenessLevel.WARNING -> WarningColor
        StalenessLevel.CRITICAL -> MaterialTheme.colorScheme.error
    }
    Text(
        text = "Updated ${formatRelativeAge(now, lastUpdatedAtMillis)}",
        style = MaterialTheme.typography.labelSmall,
        color = color,
        fontWeight = if (level == StalenessLevel.CRITICAL) FontWeight.SemiBold else null,
        modifier = modifier
    )
}
