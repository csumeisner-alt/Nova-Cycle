package com.novacycle.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.unit.dp
import androidx.compose.material3.Text
import com.novacycle.data.remote.models.PriceSnapshotResponse
import com.novacycle.ui.theme.NovaBuyGreen
import com.novacycle.ui.theme.NovaExtendedBlue
import com.novacycle.ui.theme.NovaWarningYellow

/**
 * Explains the three price references shown on the daily chart:
 * the freshest available VOO price and the prices used by each model.
 */
@Composable
fun ChartPriceSummary(
    snapshot: PriceSnapshotResponse?,
    modifier: Modifier = Modifier
) {
    if (snapshot == null) return

    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        PriceKey(NovaWarningYellow, "Current", snapshot.currentPrice)
        PriceKey(NovaBuyGreen, "Long input", snapshot.longModelPrice)
        PriceKey(NovaExtendedBlue, "Short input", snapshot.shortModelPrice)
    }
}

@Composable
private fun PriceKey(color: Color, label: String, price: Float?) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Canvas(modifier = Modifier.size(width = 8.dp, height = 3.dp)) {
            drawLine(color, center.copy(x = 0f), center.copy(x = size.width), strokeWidth = 3f)
        }
        Spacer(modifier = Modifier.width(3.dp))
        Text(
            text = if (price != null) "$label $${"%.2f".format(price)}" else "$label —",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onBackground
        )
    }
}

/**
 * Draws dashed horizontal references over a candlestick chart. The caller's
 * min/max include these prices so a current price above the last daily high
 * remains visible instead of being clipped.
 */
internal fun DrawScope.drawPriceReferenceLines(
    snapshot: PriceSnapshotResponse,
    priceMin: Float,
    priceRange: Float,
    chartHeight: Float,
    padding: Float
) {
    val references = listOf(
        // Draw the current/latest line last so it remains visible when the
        // freshest 5-minute price is also the short-model input price.
        snapshot.shortModelPrice to NovaExtendedBlue,
        snapshot.longModelPrice to NovaBuyGreen,
        snapshot.currentPrice to NovaWarningYellow
    )
    references.forEach { (price, color) ->
        if (price == null) return@forEach
        val y = chartPriceToY(price, priceMin, priceRange, chartHeight, padding)
        drawLine(
            color = color.copy(alpha = 0.8f),
            start = androidx.compose.ui.geometry.Offset(padding, y),
            end = androidx.compose.ui.geometry.Offset(size.width - padding, y),
            strokeWidth = 1.5f,
            pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 7f))
        )
    }
}

internal fun chartPriceToY(
    price: Float,
    priceMin: Float,
    priceRange: Float,
    chartHeight: Float,
    padding: Float
): Float {
    val usableH = chartHeight - padding * 2
    return chartHeight - padding - ((price - priceMin) / priceRange) * usableH
}

internal fun chartPriceBounds(
    candleMin: Float,
    candleMax: Float,
    snapshot: PriceSnapshotResponse?
): Pair<Float, Float> {
    val prices = buildList {
        add(candleMin)
        add(candleMax)
        snapshot?.currentPrice?.let(::add)
        snapshot?.longModelPrice?.let(::add)
        snapshot?.shortModelPrice?.let(::add)
    }
    val min = prices.minOrNull() ?: candleMin
    val max = prices.maxOrNull() ?: candleMax
    val margin = ((max - min) * 0.04f).coerceAtLeast(0.01f)
    return (min - margin) to (max + margin)
}