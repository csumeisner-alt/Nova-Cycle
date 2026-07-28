package com.novacycle.ui.components.confidence

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.novacycle.domain.model.ConfidencePoint
import com.novacycle.ui.theme.NovaBuyGreen
import com.novacycle.ui.theme.NovaExtendedBlue
import kotlin.math.roundToInt

/**
 * Tooltip card shown at the chart crosshair.
 *
 * Shows both plotted series and a readable timestamp. When EMA smoothing is
 * active, the plotted values ARE the EMA-smoothed values, so rows are
 * labeled accordingly instead of showing a separate (nonexistent) raw series.
 */
@Composable
fun ConfidenceTooltip(
    point: ConfidencePoint,
    emaEnabled: Boolean,
    modifier: Modifier = Modifier
) {
    val suffix = if (emaEnabled) " (EMA)" else ""
    Surface(
        modifier = modifier.width(190.dp),
        shape = MaterialTheme.shapes.small,
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.95f),
        tonalElevation = 4.dp,
        shadowElevation = 4.dp
    ) {
        Column(Modifier.padding(10.dp)) {
            Text(
                formatTooltipTimestamp(point.timestamp),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f)
            )
            Spacer(Modifier.height(6.dp))
            TooltipRow(NovaBuyGreen, "Buy (Long-term)$suffix",
                "${point.longBuyConfidence.roundToInt()}%")
            Spacer(Modifier.height(4.dp))
            TooltipRow(NovaExtendedBlue, "Buy (Short-term)$suffix",
                "${point.shortBuyConfidence.roundToInt()}%")
            if (point.isExtendedHours) {
                Spacer(Modifier.height(4.dp))
                Text(
                    "Extended hours",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
            }
        }
    }
}

@Composable
private fun TooltipRow(color: Color, label: String, value: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        androidx.compose.foundation.Canvas(Modifier.size(8.dp)) { drawCircle(color) }
        Spacer(Modifier.width(6.dp))
        Text(label, style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.weight(1f))
        Text(
            value,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}
