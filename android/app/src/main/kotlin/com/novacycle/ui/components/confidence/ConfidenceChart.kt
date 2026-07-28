package com.novacycle.ui.components.confidence

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import com.novacycle.domain.model.ConfidencePoint
import com.novacycle.ui.theme.NovaBuyGreen
import com.novacycle.ui.theme.NovaExtendedBlue
import com.novacycle.ui.theme.NovaSellRed
import com.novacycle.ui.theme.NovaWarningYellow
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt

/**
 * Premium confidence-history line chart.
 *
 * Series (data unchanged from the original chart):
 *  - Buy (Long-term)  = long-gauge buy confidence, green
 *  - Buy (Short-term) = short-gauge buy confidence, blue
 *
 * Layers: confidence zones (strong >= 70, neutral 30-70, weak <= 30),
 * 10% grid lines with tick labels, auto-scaled Y axis, axis titles,
 * extended-hours dimming, peak markers, last-crossover marker,
 * tap/drag crosshair with tooltip, and circular point markers.
 *
 * @param points        chronologically ordered confidence points
 * @param windowLabel   selected range ("3h".."6mo") — controls time-label format
 * @param emaEnabled    whether EMA smoothing is active (values are already smoothed
 *                      upstream; the tooltip labels values as EMA when enabled)
 */
@Composable
fun ConfidenceChart(
    points: List<ConfidencePoint>,
    windowLabel: String,
    emaEnabled: Boolean,
    modifier: Modifier = Modifier
) {
    if (points.size < 2) return

    var selectedIndex by remember(points) { mutableStateOf<Int?>(null) }
    var chartRect by remember { mutableStateOf<Rect?>(null) }
    var selectedPx by remember { mutableStateOf(Offset.Zero) }

    // Theme-aware colors (gold axis labels under DarkLuxe, adapts elsewhere).
    val axisLabelColor = MaterialTheme.colorScheme.primary
    val gridColor = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.15f)
    val onBg = MaterialTheme.colorScheme.onBackground

    val density = LocalDensity.current
    val yAxisPad = with(density) { 46.dp.toPx() }
    val xAxisPad = with(density) { 34.dp.toPx() }
    val topPad = with(density) { 10.dp.toPx() }
    val rightPad = with(density) { 10.dp.toPx() }

    // Auto-scaled Y range: snap to 10% boundaries around the data, min span 20.
    val (yMin, yMax) = remember(points) {
        val values = points.flatMap { listOf(it.longBuyConfidence, it.shortBuyConfidence) }
        var lo = floor((values.min() / 10f)) * 10f - 10f
        var hi = ceil((values.max() / 10f)) * 10f + 10f
        lo = lo.coerceIn(0f, 90f)
        hi = hi.coerceIn(10f, 100f)
        if (hi - lo < 20f) { lo = (hi - 20f).coerceAtLeast(0f); hi = (lo + 20f).coerceAtMost(100f) }
        lo to hi
    }

    val longPeakIdx = remember(points) { points.indices.maxBy { points[it].longBuyConfidence } }
    val shortPeakIdx = remember(points) { points.indices.maxBy { points[it].shortBuyConfidence } }
    val crossoverIdx = remember(points) { lastCrossoverIndex(points) }

    Box(modifier = modifier) {
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(points) {
                    detectTapGestures(onTap = { pos ->
                        selectedIndex = nearestIndex(pos.x, yAxisPad, size.width - rightPad, points.size)
                    })
                }
                .pointerInput(points) {
                    // Horizontal-only drag: vertical drags must stay available
                    // to the enclosing PullRefreshBox for pull-to-refresh.
                    detectHorizontalDragGestures(
                        onHorizontalDrag = { change, _ ->
                            change.consume()
                            selectedIndex =
                                nearestIndex(change.position.x, yAxisPad, size.width - rightPad, points.size)
                        }
                    )
                }
        ) {
            val plotLeft = yAxisPad
            val plotRight = size.width - rightPad
            val plotTop = topPad
            val plotBottom = size.height - xAxisPad
            val plotW = plotRight - plotLeft
            val plotH = plotBottom - plotTop
            chartRect = Rect(plotLeft, plotTop, plotRight, plotBottom)

            val count = points.size
            val stepX = plotW / (count - 1).coerceAtLeast(1)
            fun xAt(i: Int) = plotLeft + i * stepX
            fun yAt(v: Float) = plotBottom - ((v - yMin) / (yMax - yMin)).coerceIn(0f, 1f) * plotH

            // ── Confidence zones ─────────────────────────────────────────
            fun zone(fromV: Float, toV: Float, color: Color) {
                val top = yAt(toV.coerceIn(yMin, yMax))
                val bottom = yAt(fromV.coerceIn(yMin, yMax))
                if (bottom > top) drawRect(color, Offset(plotLeft, top),
                    androidx.compose.ui.geometry.Size(plotW, bottom - top))
            }
            if (yMax > 70f) zone(70f, yMax, NovaBuyGreen.copy(alpha = 0.06f))     // strong
            if (yMin < 30f) zone(yMin, 30f, NovaSellRed.copy(alpha = 0.06f))      // weak
            zone(maxOf(30f, yMin), minOf(70f, yMax), onBg.copy(alpha = 0.02f))    // neutral

            // ── Grid lines every 10% + tick labels ───────────────────────
            val tickPaint = android.graphics.Paint().apply {
                color = axisLabelColor.toArgb()
                textSize = with(this@Canvas) { 10.dp.toPx() }
                isAntiAlias = true
                textAlign = android.graphics.Paint.Align.RIGHT
            }
            var level = ceil(yMin / 10f) * 10f
            while (level <= yMax) {
                val y = yAt(level)
                drawLine(gridColor, Offset(plotLeft, y), Offset(plotRight, y), 1f)
                drawContext.canvas.nativeCanvas.drawText(
                    "${level.roundToInt()}", plotLeft - 6.dp.toPx(), y + 4.dp.toPx(), tickPaint
                )
                level += 10f
            }

            // ── X-axis time labels (format adapts to the range) ──────────
            val xPaint = android.graphics.Paint(tickPaint).apply {
                textAlign = android.graphics.Paint.Align.CENTER
            }
            val labelCount = 4
            for (l in 0 until labelCount) {
                val i = (l * (count - 1) / (labelCount - 1))
                drawContext.canvas.nativeCanvas.drawText(
                    formatTimeLabel(points[i].timestamp, windowLabel),
                    xAt(i).coerceIn(plotLeft + 20.dp.toPx(), plotRight - 20.dp.toPx()),
                    plotBottom + 14.dp.toPx(), xPaint
                )
            }

            // ── Axis titles ──────────────────────────────────────────────
            val titlePaint = android.graphics.Paint(tickPaint).apply {
                textAlign = android.graphics.Paint.Align.CENTER
                textSize = with(this@Canvas) { 11.dp.toPx() }
            }
            drawContext.canvas.nativeCanvas.drawText(
                "Time", (plotLeft + plotRight) / 2f, size.height - 4.dp.toPx(), titlePaint
            )
            drawContext.canvas.nativeCanvas.apply {
                save()
                rotate(-90f, 12.dp.toPx(), (plotTop + plotBottom) / 2f)
                drawText("Confidence (%)", 12.dp.toPx(), (plotTop + plotBottom) / 2f + 4.dp.toPx(), titlePaint)
                restore()
            }

            // ── Gradient fills under each line ───────────────────────────
            fun fillUnder(values: List<Float>, color: Color, topAlpha: Float) {
                val path = Path().apply {
                    moveTo(plotLeft, plotBottom)
                    values.forEachIndexed { i, v -> lineTo(xAt(i), yAt(v)) }
                    lineTo(plotRight, plotBottom)
                    close()
                }
                drawPath(path, brush = Brush.verticalGradient(
                    colors = listOf(color.copy(alpha = topAlpha), color.copy(alpha = 0f)),
                    startY = plotTop, endY = plotBottom
                ))
            }
            fillUnder(points.map { it.shortBuyConfidence }, NovaExtendedBlue, 0.12f)
            fillUnder(points.map { it.longBuyConfidence }, NovaBuyGreen, 0.18f)

            // ── Series lines (extended-hours segments dimmed) ────────────
            for (i in 1 until count) {
                val alpha = if (points[i].isExtendedHours) 0.35f else 0.9f
                drawLine(NovaBuyGreen.copy(alpha = alpha),
                    Offset(xAt(i - 1), yAt(points[i - 1].longBuyConfidence)),
                    Offset(xAt(i), yAt(points[i].longBuyConfidence)),
                    strokeWidth = 2.5f, cap = StrokeCap.Round)
                drawLine(NovaExtendedBlue.copy(alpha = alpha),
                    Offset(xAt(i - 1), yAt(points[i - 1].shortBuyConfidence)),
                    Offset(xAt(i), yAt(points[i].shortBuyConfidence)),
                    strokeWidth = 2.5f, cap = StrokeCap.Round)
            }

            // 50%-cross markers on the long-term line (existing behavior).
            for (i in 1 until count) {
                val prev = points[i - 1].longBuyConfidence
                val curr = points[i].longBuyConfidence
                if ((prev < 50f && curr >= 50f) || (prev >= 50f && curr < 50f)) {
                    drawLine(NovaWarningYellow.copy(alpha = 0.5f),
                        Offset(xAt(i), plotTop), Offset(xAt(i), plotBottom), 1.5f)
                }
            }

            // ── Peak + crossover markers ─────────────────────────────────
            fun peakMarker(i: Int, v: Float, color: Color) {
                drawCircle(color, radius = 5.dp.toPx(), center = Offset(xAt(i), yAt(v)),
                    style = Stroke(width = 2.dp.toPx()))
            }
            peakMarker(longPeakIdx, points[longPeakIdx].longBuyConfidence, NovaBuyGreen)
            peakMarker(shortPeakIdx, points[shortPeakIdx].shortBuyConfidence, NovaExtendedBlue)
            crossoverIdx?.let { i ->
                val y = yAt((points[i].longBuyConfidence + points[i].shortBuyConfidence) / 2f)
                drawCircle(NovaWarningYellow, radius = 4.dp.toPx(), center = Offset(xAt(i), y))
                drawCircle(NovaWarningYellow.copy(alpha = 0.4f), radius = 8.dp.toPx(),
                    center = Offset(xAt(i), y), style = Stroke(width = 1.5f))
            }

            // ── Crosshair + point markers at the selection ───────────────
            selectedIndex?.let { i ->
                val x = xAt(i)
                selectedPx = Offset(x, plotTop)
                drawLine(onBg.copy(alpha = 0.6f), Offset(x, plotTop), Offset(x, plotBottom),
                    strokeWidth = 1.5f,
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 6f)))
                drawCircle(NovaBuyGreen, 4.dp.toPx(), Offset(x, yAt(points[i].longBuyConfidence)))
                drawCircle(NovaExtendedBlue, 4.dp.toPx(), Offset(x, yAt(points[i].shortBuyConfidence)))
            }
        }

        // Tooltip anchored near the crosshair, kept inside the chart bounds.
        selectedIndex?.let { i ->
            val rect = chartRect ?: return@let
            val tooltipWidthPx = with(density) { 190.dp.toPx() }
            val flip = selectedPx.x + tooltipWidthPx + 16 > rect.right
            val xOff = if (flip) selectedPx.x - tooltipWidthPx - 12 else selectedPx.x + 12
            ConfidenceTooltip(
                point = points[i],
                emaEnabled = emaEnabled,
                modifier = Modifier.offset {
                    IntOffset(xOff.roundToInt().coerceAtLeast(0), (rect.top + 8).roundToInt())
                }
            )
        }
    }
}

/** Map a raw x pixel to the nearest data index within the plot area. */
private fun nearestIndex(x: Float, plotLeft: Float, plotRight: Float, count: Int): Int {
    val frac = ((x - plotLeft) / (plotRight - plotLeft)).coerceIn(0f, 1f)
    return (frac * (count - 1)).roundToInt().coerceIn(0, count - 1)
}

/** Last index where the long-term and short-term lines cross, or null. */
internal fun lastCrossoverIndex(points: List<ConfidencePoint>): Int? {
    for (i in points.size - 1 downTo 1) {
        val prevDiff = points[i - 1].longBuyConfidence - points[i - 1].shortBuyConfidence
        val currDiff = points[i].longBuyConfidence - points[i].shortBuyConfidence
        if (prevDiff == 0f) continue
        if (prevDiff * currDiff <= 0f) return i
    }
    return null
}

/** Format an ISO timestamp for the X axis based on the selected range. */
internal fun formatTimeLabel(isoTimestamp: String, window: String): String = try {
    val dt = LocalDateTime.parse(isoTimestamp.removeSuffix("Z").substringBefore("+"))
    val pattern = when {
        window.endsWith("h") -> "HH:mm"
        window == "7d" -> "EEE HH:mm"
        window.endsWith("mo") -> "MMM d"
        else -> "MMM d"
    }
    dt.format(DateTimeFormatter.ofPattern(pattern))
} catch (e: Exception) {
    isoTimestamp.take(10)
}

/** Full readable timestamp for the tooltip. */
internal fun formatTooltipTimestamp(isoTimestamp: String): String = try {
    val dt = LocalDateTime.parse(isoTimestamp.removeSuffix("Z").substringBefore("+"))
    dt.format(DateTimeFormatter.ofPattern("MMM d, yyyy  HH:mm"))
} catch (e: Exception) {
    isoTimestamp
}
