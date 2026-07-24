package com.novacycle.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

/**
 * ConfidenceRibbon — compact bar chart showing confidence momentum over time.
 *
 * Momentum = confidence(t) - confidence(t-1).
 * Positive momentum: green bars. Negative momentum: red bars.
 * Used at the bottom of the FilteredChartScreen.
 */
@Composable
fun ConfidenceRibbon(
    momentumPoints: List<Float>,
    modifier: Modifier = Modifier
) {
    if (momentumPoints.isEmpty()) {
        Box(modifier.height(48.dp)) {
            Text(
                "No confidence data",
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFF616161)
            )
        }
        return
    }

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(48.dp)
    ) {
        val w = size.width
        val h = size.height
        val midY = h / 2f

        if (momentumPoints.isEmpty()) return@Canvas

        val barWidth = (w / momentumPoints.size.coerceAtLeast(1)) * 0.7f
        val maxMom = 0.3f  // Clamp momentum display to ±0.3

        momentumPoints.forEachIndexed { i, mom ->
            val x = (i.toFloat() / momentumPoints.size) * w + barWidth * 0.15f
            val clampedMom = mom.coerceIn(-maxMom, maxMom)
            val barH = (kotlin.math.abs(clampedMom) / maxMom) * (h / 2f)
            val color = if (mom >= 0) Color(0xFF00C853) else Color(0xFFD50000)

            if (mom >= 0) {
                // Positive: bar goes up from midY
                drawRect(
                    color = color.copy(alpha = 0.7f),
                    topLeft = Offset(x, midY - barH),
                    size = Size(barWidth, barH)
                )
            } else {
                // Negative: bar goes down from midY
                drawRect(
                    color = color.copy(alpha = 0.7f),
                    topLeft = Offset(x, midY),
                    size = Size(barWidth, barH)
                )
            }
        }

        // Center baseline
        drawLine(
            color = Color(0xFF424242),
            start = Offset(0f, midY),
            end = Offset(w, midY),
            strokeWidth = 1.dp.toPx()
        )
    }
}
