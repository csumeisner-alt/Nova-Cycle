package com.novacycle.ui.components.confidence

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.novacycle.ui.theme.NovaBuyGreen
import com.novacycle.ui.theme.NovaExtendedBlue

/**
 * Legend for the confidence-history chart.
 * Terminology: both lines plot BUY confidence — one per gauge horizon.
 */
@Composable
fun ConfidenceLegend(modifier: Modifier = Modifier) {
    Row(modifier = modifier, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        LegendLine(NovaBuyGreen, "Buy (Long-term)")
        LegendLine(NovaExtendedBlue, "Buy (Short-term)")
        LegendLine(NovaExtendedBlue.copy(alpha = 0.35f), "Extended Hours")
    }
}

@Composable
private fun LegendLine(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Canvas(modifier = Modifier.size(12.dp, 3.dp)) { drawRect(color) }
        Spacer(Modifier.width(4.dp))
        Text(label, style = MaterialTheme.typography.labelSmall)
    }
}
