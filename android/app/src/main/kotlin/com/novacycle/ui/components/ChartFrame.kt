package com.novacycle.ui.components

import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.novacycle.data.remote.models.CandleResponse
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.pow

/** How the price series is rendered on the chart. */
enum class ChartRenderMode { CANDLES, LINE }

/** Reserved space for the right price axis, in dp. */
val PRICE_AXIS_WIDTH = 52.dp

/** Reserved space for the bottom time axis, in dp. */
val TIME_AXIS_HEIGHT = 18.dp

private val MARKET_ZONE: ZoneId = ZoneId.of("America/New_York")
private val TIME_FMT = DateTimeFormatter.ofPattern("HH:mm")
private val DATE_FMT = DateTimeFormatter.ofPattern("MMM d")
private val DATETIME_FMT = DateTimeFormatter.ofPattern("MMM d · HH:mm")

/**
 * Parse a backend candle timestamp (UTC-naive ISO string) to epoch millis.
 * Returns null when the string is malformed rather than throwing mid-draw.
 */
fun parseCandleTimestampMillis(timestamp: String): Long? = try {
    LocalDateTime.parse(timestamp).toInstant(ZoneOffset.UTC).toEpochMilli()
} catch (_: Exception) {
    null
}

/** Format a candle timestamp as a short ET time-of-day label ("14:35"). */
fun formatCandleTime(timestamp: String): String = formatCandle(timestamp, TIME_FMT)

/** Format a candle timestamp as a short ET date label ("Jul 31"). */
fun formatCandleDate(timestamp: String): String = formatCandle(timestamp, DATE_FMT)

/** Format a candle timestamp as a full ET date-time label ("Jul 31 · 14:35"). */
fun formatCandleDateTime(timestamp: String): String = formatCandle(timestamp, DATETIME_FMT)

private fun formatCandle(timestamp: String, fmt: DateTimeFormatter): String = try {
    LocalDateTime.parse(timestamp)
        .atZone(ZoneOffset.UTC)
        .withZoneSameInstant(MARKET_ZONE)
        .format(fmt)
} catch (_: Exception) {
    timestamp
}

/**
 * Compute "nice" evenly spaced tick values covering [min, max].
 * Steps are 1/2/5×10^n so labels read like a pro terminal ($680.00, $682.50…).
 */
fun niceTicks(min: Float, max: Float, targetCount: Int = 5): List<Float> {
    if (max <= min || targetCount < 2) return emptyList()
    val rawStep = (max - min) / targetCount
    val magnitude = 10.0.pow(floor(log10(rawStep.toDouble()))).toFloat()
    val residual = rawStep / magnitude
    val step = when {
        residual > 5f -> 10f
        residual > 2f -> 5f
        residual > 1f -> 2f
        else -> 1f
    } * magnitude
    val first = ceil(min / step) * step
    val ticks = mutableListOf<Float>()
    var v = first
    while (v <= max + step * 0.001f) {
        ticks.add(v)
        v += step
    }
    return ticks
}

/**
 * Draw the right-side price axis: horizontal gridlines across the plot area
 * plus rounded price labels in the reserved axis gutter.
 */
fun DrawScope.drawPriceAxis(
    priceMin: Float,
    priceRange: Float,
    plotWidth: Float,
    plotHeight: Float,
    padding: Float,
    labelColor: Color,
    gridColor: Color,
    textSizePx: Float
) {
    val ticks = niceTicks(priceMin, priceMin + priceRange)
    val paint = android.graphics.Paint().apply {
        color = labelColor.toArgb()
        textSize = textSizePx
        isAntiAlias = true
    }
    ticks.forEach { price ->
        val y = chartPriceToY(price, priceMin, priceRange, plotHeight, padding)
        if (y < padding || y > plotHeight - padding) return@forEach
        drawLine(
            color = gridColor,
            start = Offset(padding, y),
            end = Offset(plotWidth, y),
            strokeWidth = 1f
        )
        drawContext.canvas.nativeCanvas.drawText(
            "%.2f".format(price),
            plotWidth + 6f,
            y + textSizePx * 0.35f,
            paint
        )
    }
}

/**
 * Draw adaptive bottom time labels for the currently visible candles.
 * Intraday timeframes show ET clock times; daily shows dates. Label density
 * adapts to bar width so labels never collide, and pan/zoom stay in sync
 * because x positions use the same barWidth/offset math as the candles.
 */
fun DrawScope.drawTimeAxis(
    candles: List<CandleResponse>,
    barWidth: Float,
    offsetX: Float,
    padding: Float,
    plotWidth: Float,
    plotHeight: Float,
    isIntraday: Boolean,
    labelColor: Color,
    textSizePx: Float
) {
    if (candles.isEmpty()) return
    val paint = android.graphics.Paint().apply {
        color = labelColor.toArgb()
        textSize = textSizePx
        isAntiAlias = true
        textAlign = android.graphics.Paint.Align.CENTER
    }
    val minLabelSpacingPx = textSizePx * 5f
    val stride = ceil(minLabelSpacingPx / barWidth).toInt().coerceAtLeast(1)
    val y = plotHeight + textSizePx + 2f
    var lastDayLabel: String? = null
    for (i in candles.indices step stride) {
        val x = i * barWidth + offsetX + padding + barWidth / 2
        if (x < padding || x > plotWidth) continue
        val label = if (isIntraday) {
            // Show the date instead of the time when the trading day changes.
            val day = formatCandleDate(candles[i].timestamp)
            if (day != lastDayLabel) { lastDayLabel = day; day } else formatCandleTime(candles[i].timestamp)
        } else {
            formatCandleDate(candles[i].timestamp)
        }
        drawContext.canvas.nativeCanvas.drawText(label, x, y, paint)
    }
}

/**
 * Draw the live last-price marker: a solid horizontal line at the freshest
 * price with a filled price tag pinned into the axis gutter.
 */
fun DrawScope.drawLastPriceMarker(
    price: Float,
    priceMin: Float,
    priceRange: Float,
    plotWidth: Float,
    plotHeight: Float,
    padding: Float,
    color: Color,
    textSizePx: Float
) {
    val y = chartPriceToY(price, priceMin, priceRange, plotHeight, padding)
    drawLine(
        color = color.copy(alpha = 0.9f),
        start = Offset(padding, y),
        end = Offset(plotWidth, y),
        strokeWidth = 2f
    )
    val tagArgb = color.toArgb()
    val tagPaint = android.graphics.Paint()
    tagPaint.color = tagArgb
    tagPaint.isAntiAlias = true
    val textPaint = android.graphics.Paint()
    textPaint.color = android.graphics.Color.WHITE
    textPaint.textSize = textSizePx
    textPaint.isAntiAlias = true
    textPaint.isFakeBoldText = true
    val tagHeight = textSizePx * 1.5f
    drawContext.canvas.nativeCanvas.apply {
        drawRoundRect(
            plotWidth + 2f, y - tagHeight / 2,
            size.width - 2f, y + tagHeight / 2,
            4f, 4f, tagPaint
        )
        drawText("%.2f".format(price), plotWidth + 6f, y + textSizePx * 0.35f, textPaint)
    }
}

/**
 * Draw the close-price line (LINE render mode) with a subtle gradient fill.
 * Uses the same scales as the candles so overlays remain aligned.
 */
fun DrawScope.drawCloseLine(
    candles: List<CandleResponse>,
    barWidth: Float,
    offsetX: Float,
    padding: Float,
    priceMin: Float,
    priceRange: Float,
    plotHeight: Float,
    lineColor: Color
) {
    if (candles.size < 2) return
    val line = Path()
    val fill = Path()
    var started = false
    var firstX = 0f
    var lastX = 0f
    candles.forEachIndexed { i, c ->
        val x = i * barWidth + offsetX + padding + barWidth / 2
        val y = chartPriceToY(c.close, priceMin, priceRange, plotHeight, padding)
        if (!started) {
            line.moveTo(x, y); fill.moveTo(x, y); firstX = x; started = true
        } else {
            line.lineTo(x, y); fill.lineTo(x, y)
        }
        lastX = x
    }
    fill.lineTo(lastX, plotHeight - padding)
    fill.lineTo(firstX, plotHeight - padding)
    fill.close()
    drawPath(
        fill,
        Brush.verticalGradient(
            colors = listOf(lineColor.copy(alpha = 0.25f), lineColor.copy(alpha = 0.0f)),
            startY = padding,
            endY = plotHeight - padding
        )
    )
    drawPath(line, lineColor, style = Stroke(width = 3f))
}

/** Draw the crosshair lines through the selected candle. */
fun DrawScope.drawCrosshair(
    index: Int,
    candle: CandleResponse,
    barWidth: Float,
    offsetX: Float,
    padding: Float,
    priceMin: Float,
    priceRange: Float,
    plotWidth: Float,
    plotHeight: Float,
    color: Color
) {
    val x = index * barWidth + offsetX + padding + barWidth / 2
    val y = chartPriceToY(candle.close, priceMin, priceRange, plotHeight, padding)
    val dash = PathEffect.dashPathEffect(floatArrayOf(6f, 6f))
    drawLine(color, Offset(x, 0f), Offset(x, plotHeight), 1.5f, pathEffect = dash)
    drawLine(color, Offset(padding, y), Offset(plotWidth, y), 1.5f, pathEffect = dash)
}

/**
 * Long-press crosshair gesture: activates on long-press, tracks the finger
 * while dragging, and clears when the finger lifts. Reports the raw x
 * position; the caller maps it to a candle index with [candleIndexAt].
 */
fun Modifier.chartCrosshairInput(
    key: Any?,
    onCrosshair: (x: Float, active: Boolean) -> Unit
): Modifier = pointerInput(key) {
    detectDragGesturesAfterLongPress(
        onDragStart = { onCrosshair(it.x, true) },
        onDrag = { change, _ ->
            change.consume()
            onCrosshair(change.position.x, true)
        },
        onDragEnd = { onCrosshair(0f, false) },
        onDragCancel = { onCrosshair(0f, false) }
    )
}

/**
 * Map a touch x position back to a candle index, honoring pan/zoom.
 * Returns null when the touch falls outside the candle series.
 */
fun candleIndexAt(
    touchX: Float,
    barWidth: Float,
    offsetX: Float,
    padding: Float,
    candleCount: Int
): Int? {
    val idx = floor((touchX - offsetX - padding) / barWidth).toInt()
    return if (idx in 0 until candleCount) idx else null
}

/**
 * Attach signals to candle indices even when the chart timeframe is coarser
 * than the signal timestamps: each signal maps to the last candle whose
 * bucket starts at or before the signal time.
 */
fun signalIndexByCandle(
    candles: List<CandleResponse>,
    signalTimestamps: List<String>
): Map<String, Int> {
    val candleMillis = candles.map { parseCandleTimestampMillis(it.timestamp) ?: Long.MIN_VALUE }
    val out = mutableMapOf<String, Int>()
    signalTimestamps.forEach { ts ->
        val sMillis = parseCandleTimestampMillis(ts) ?: return@forEach
        var lo = 0
        var hi = candleMillis.size - 1
        var best = -1
        while (lo <= hi) {
            val mid = (lo + hi) / 2
            if (candleMillis[mid] <= sMillis) { best = mid; lo = mid + 1 } else hi = mid - 1
        }
        if (best >= 0) out[ts] = best
    }
    return out
}

/**
 * Crosshair readout card: exact ET time plus OHLC + volume of the candle
 * under the finger. Overlaid at the top-start of the chart box.
 */
@Composable
fun CrosshairReadout(candle: CandleResponse, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier.padding(8.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f),
        shape = MaterialTheme.shapes.small,
        tonalElevation = 4.dp
    ) {
        Column(Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
            Text(
                formatCandleDateTime(candle.timestamp) + " ET" +
                    if (candle.isExtendedHours) " · ext" else "",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ReadoutValue("O", candle.open)
                ReadoutValue("H", candle.high)
                ReadoutValue("L", candle.low)
                ReadoutValue("C", candle.close)
            }
            Text(
                "Vol ${formatVolume(candle.volume)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun ReadoutValue(label: String, value: Float) {
    Text(
        "$label ${"%.2f".format(value)}",
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurface
    )
}

private fun formatVolume(volume: Long): String = when {
    abs(volume) >= 1_000_000 -> "%.1fM".format(volume / 1_000_000f)
    abs(volume) >= 1_000 -> "%.1fK".format(volume / 1_000f)
    else -> volume.toString()
}

/**
 * Chart header freshness line: last-candle ET timestamp and session, colored
 * by market-aware staleness so an old feed is impossible to miss.
 */
@Composable
fun ChartFreshnessHeader(
    lastCandle: CandleResponse?,
    modifier: Modifier = Modifier
) {
    if (lastCandle == null) return
    val nowMillis = rememberTickingNow()
    val candleMillis = parseCandleTimestampMillis(lastCandle.timestamp)
    val level = if (candleMillis != null) {
        marketAwareStalenessLevel(
            nowMillis, candleMillis,
            DEFAULT_WARNING_THRESHOLD_MILLIS, DEFAULT_CRITICAL_THRESHOLD_MILLIS
        )
    } else StalenessLevel.CRITICAL
    val color = when (level) {
        StalenessLevel.FRESH -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.75f)
        StalenessLevel.WARNING -> Color(0xFFB26A00)
        StalenessLevel.CRITICAL -> MaterialTheme.colorScheme.error
    }
    val session = when (lastCandle.sessionType) {
        "pre_market" -> "Pre-market"
        "after_hours" -> "After-hours"
        else -> "Regular"
    }
    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        Text(
            "Last candle ${formatCandleDateTime(lastCandle.timestamp)} ET · $session" +
                if (level != StalenessLevel.FRESH) " · STALE" else "",
            style = MaterialTheme.typography.labelSmall,
            color = color,
            fontWeight = if (level == StalenessLevel.CRITICAL) FontWeight.SemiBold else null
        )
    }
}
