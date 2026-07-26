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
 *
 * Outside US market hours (nights, weekends, holidays) data legitimately stops
 * updating, so staleness is measured against the last market close instead of
 * the wall clock — see [marketAwareStalenessLevel].
 *
 * When the market is closed and data is up to date as of the last close, the
 * label shows e.g. "Updated at Fri close · Market closed" so users know the
 * pause is intentional and not a connection problem.
 *
 * @param extendedHoursAware When `true`, the "Market closed" hint is suppressed
 * during pre-market (04:00–09:30 ET) and after-hours (16:00–20:00 ET) sessions
 * on regular trading days. Callers that surface extended-hours data should set
 * this to `true` so the label does not mislead users who are active before the
 * regular open or after the regular close.
 */
@Composable
fun UpdatedAgoLabel(
    lastUpdatedAtMillis: Long?,
    modifier: Modifier = Modifier,
    warningThresholdMillis: Long = DEFAULT_WARNING_THRESHOLD_MILLIS,
    criticalThresholdMillis: Long = DEFAULT_CRITICAL_THRESHOLD_MILLIS,
    extendedHoursAware: Boolean = false
) {
    if (lastUpdatedAtMillis == null) return
    val now = rememberTickingNow()
    val level = marketAwareStalenessLevel(now, lastUpdatedAtMillis, warningThresholdMillis, criticalThresholdMillis)
    val color = when (level) {
        StalenessLevel.FRESH -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        StalenessLevel.WARNING -> WarningColor
        StalenessLevel.CRITICAL -> MaterialTheme.colorScheme.error
    }

    val marketOpen = MarketHours.isMarketOpen(now)
    // Suppress the "Market closed" hint during extended-hours windows when the
    // caller has opted in, since pre-market and after-hours are active sessions.
    val inExtendedHours = extendedHoursAware && MarketHours.isExtendedHours(now)
    val text = if (!marketOpen && !inExtendedHours && level == StalenessLevel.FRESH) {
        // Data is current as of the last close — tell the user the market is closed
        // rather than showing a confusingly large relative age.
        val day = MarketHours.lastSessionCloseDayLabel(now)
        "Updated at $day close · Market closed"
    } else {
        "Updated ${formatRelativeAge(now, lastUpdatedAtMillis)}"
    }

    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        color = color,
        fontWeight = if (level == StalenessLevel.CRITICAL) FontWeight.SemiBold else null,
        modifier = modifier
    )
}
