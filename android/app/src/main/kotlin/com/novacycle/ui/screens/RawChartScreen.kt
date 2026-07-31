package com.novacycle.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import com.novacycle.ui.components.PullRefreshBox
import com.novacycle.ui.components.SignalStoryCard
import com.novacycle.ui.components.UpdatedAgoLabel
import com.novacycle.ui.components.ChartPriceSummary
import com.novacycle.ui.components.chartPriceBounds
import com.novacycle.ui.components.chartPriceToY
import com.novacycle.ui.components.drawPriceReferenceLines
import com.novacycle.ui.theme.*
import com.novacycle.viewmodel.RawChartViewModel
import com.novacycle.viewmodel.SettingsViewModel
import kotlin.math.abs

/**
 * Raw Chart screen — shows ALL signals overlaid on a candlestick chart.
 *
 * Signal marker legend:
 *  ▲ Green filled    = Regular BUY
 *  ▼ Red filled      = Regular SELL
 *  △ Blue hollow     = Extended-hours BUY
 *  ▽ Blue hollow     = Extended-hours SELL
 *  ◆ Purple diamond  = Gap-driven signal
 *  ▲ Gray faded      = Liquidity-filtered signal (shown dimmed)
 *  ▲ Yellow triangle = Macro-override suppressed
 *
 * Chart supports pinch-to-zoom via transformable modifier.
 */
@Composable
fun RawChartScreen(
    viewModel: RawChartViewModel = hiltViewModel(),
    settingsViewModel: SettingsViewModel = hiltViewModel()
) {
    val uiState  by viewModel.uiState.collectAsStateWithLifecycle()
    val settings by settingsViewModel.settings.collectAsStateWithLifecycle()

    LaunchedEffect(settings) { viewModel.applySettings(settings) }

    var selectedSignal by remember { mutableStateOf<SignalData?>(null) }
    val windows = listOf("7d", "30d", "90d")

    PullRefreshBox(
        refreshing = uiState.isLoading,
        onRefresh = { viewModel.loadData() },
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
    Column(modifier = Modifier.fillMaxSize()) {
        // Toolbar
        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Raw Chart", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Text("VOO", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
        }

        // "Updated X ago" freshness label, ticking as time passes
        UpdatedAgoLabel(
            lastUpdatedAtMillis = uiState.lastUpdatedAtMillis,
            modifier = Modifier.padding(horizontal = 12.dp),
            extendedHoursAware = true
        )

        // Window selector
        Row(modifier = Modifier.padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            windows.forEach { w ->
                FilterChip(
                    selected = uiState.selectedWindow == w,
                    onClick  = { viewModel.setWindow(w) },
                    label    = { Text(w) }
                )
            }
        }

        if (uiState.isLoading) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        if (uiState.error != null) {
            Text("⚠️ ${uiState.error}", color = NovaSellRed,
                style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(12.dp))
        }

        ChartPriceSummary(
            snapshot = uiState.priceSnapshot,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
        )

        // Chart
        if (uiState.candles.isNotEmpty()) {
            CandlestickChart(
                candles = uiState.candles,
                signals = uiState.signals,
                priceSnapshot = uiState.priceSnapshot,
                modifier = Modifier.fillMaxWidth().weight(1f),
                onSignalTapped = { selectedSignal = it }
            )
        } else if (!uiState.isLoading) {
            Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
                Text("No chart data available", color = NovaNeutralGray)
            }
        }

        // Legend
        ChartLegend(modifier = Modifier.padding(12.dp))
    }
    }

    // Signal story bottom sheet
    selectedSignal?.let { signal ->
        SignalStoryCard(
            signal     = signal,
            storyLevel = settings.storyCardLevel,
            onDismiss  = { selectedSignal = null }
        )
    }
}

/**
 * Canvas candlestick chart with signal overlays.
 * Zoom: pinch via transformable modifier. Pan: drag via transformable.
 */
@Composable
fun CandlestickChart(
    candles: List<CandleResponse>,
    signals: List<SignalData>,
    priceSnapshot: com.novacycle.data.remote.models.PriceSnapshotResponse? = null,
    modifier: Modifier = Modifier,
    onSignalTapped: (SignalData) -> Unit = {}
) {
    if (candles.isEmpty()) return

    var scale   by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }

    val transformableState = rememberTransformableState { zoomChange, panChange, _ ->
        scale   = (scale * zoomChange).coerceIn(0.5f, 8f)
        offsetX = (offsetX + panChange.x).coerceIn(-8000f, 0f)
    }

    val (priceMin, boundedPriceMax) = chartPriceBounds(
        candles.minOf { it.low },
        candles.maxOf { it.high },
        priceSnapshot
    )
    val priceMax   = boundedPriceMax
    val priceRange = (priceMax - priceMin).coerceAtLeast(0.01f)

    Box(modifier = modifier.transformable(state = transformableState)) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val padding     = 8.dp.toPx()
            val candleCount = candles.size
            val barWidth    = ((size.width / candleCount) * scale).coerceAtLeast(4f)
            val wickWidth   = (barWidth * 0.15f).coerceAtLeast(1f)

            val signalByTimestamp = signals.associateBy { it.timestamp }

            // Vertical session separators at every session-type transition
            // (pre-market → regular → after-hours), drawn behind the candles.
            drawSessionSeparators(candles, barWidth, offsetX, padding)
            priceSnapshot?.let {
                drawPriceReferenceLines(it, priceMin, priceRange, size.height, padding)
            }

            candles.forEachIndexed { index, candle ->
                val x = index * barWidth + offsetX + padding
                if (x + barWidth < 0 || x > size.width) return@forEachIndexed

                val openY  = priceToY(candle.open,  priceMin, priceRange, size.height, padding)
                val closeY = priceToY(candle.close, priceMin, priceRange, size.height, padding)
                val highY  = priceToY(candle.high,  priceMin, priceRange, size.height, padding)
                val lowY   = priceToY(candle.low,   priceMin, priceRange, size.height, padding)

                val isBullish   = candle.close >= candle.open
                val candleColor = when {
                    candle.isExtendedHours -> NovaExtendedBlue.copy(alpha = 0.7f)
                    isBullish              -> NovaBuyGreen
                    else                   -> NovaSellRed
                }

                // Wick
                drawLine(
                    color  = candleColor,
                    start  = Offset(x + barWidth / 2, highY),
                    end    = Offset(x + barWidth / 2, lowY),
                    strokeWidth = wickWidth
                )
                // Body
                drawRect(
                    color   = candleColor,
                    topLeft = Offset(x + wickWidth, minOf(openY, closeY)),
                    size    = Size((barWidth - wickWidth * 2).coerceAtLeast(1f), abs(openY - closeY).coerceAtLeast(1f))
                )

                // Signal marker
                val signal = signalByTimestamp[candle.timestamp]
                if (signal != null) {
                    val markerSize = (barWidth * 0.8f).coerceIn(8f, 20f)
                    drawSignalMarker(signal, x + barWidth / 2, highY, lowY, markerSize)
                }
            }
        }
    }
}

/**
 * Draws a subtle dashed vertical line wherever consecutive candles change
 * session type (e.g. pre_market → regular), so session boundaries are visible
 * on the chart without overpowering the candles.
 */
internal fun DrawScope.drawSessionSeparators(
    candles: List<CandleResponse>,
    barWidth: Float,
    offsetX: Float,
    padding: Float
) {
    val dash = PathEffect.dashPathEffect(floatArrayOf(8f, 8f))
    for (i in 1 until candles.size) {
        if (candles[i].sessionType != candles[i - 1].sessionType) {
            val x = i * barWidth + offsetX + padding
            if (x < 0 || x > size.width) continue
            drawLine(
                color = NovaExtendedBlue.copy(alpha = 0.35f),
                start = Offset(x, 0f),
                end   = Offset(x, size.height),
                strokeWidth = 1.5f,
                pathEffect  = dash
            )
        }
    }
}

private fun DrawScope.drawSignalMarker(
    signal: SignalData,
    centerX: Float,
    candleHighY: Float,
    candleLowY: Float,
    markerSize: Float
) {
    val isBuy = signal.isBuy
    val gap   = markerSize * 0.5f
    val cy    = if (isBuy) candleLowY + gap + markerSize else candleHighY - gap - markerSize

    when {
        signal.gapType != null -> drawDiamond(Offset(centerX, cy), markerSize, NovaGapPurple)
        signal.macroOverrideApplied -> drawTriangle(Offset(centerX, cy), markerSize, isBuy, NovaWarningYellow, filled = true)
        signal.liquidityScore < 0.5f -> drawTriangle(Offset(centerX, cy), markerSize, isBuy, NovaFadedGray, filled = true)
        signal.isExtendedHours -> drawTriangle(Offset(centerX, cy), markerSize, isBuy, NovaExtendedBlue, filled = false)
        else -> drawTriangle(Offset(centerX, cy), markerSize, isBuy, if (isBuy) NovaBuyGreen else NovaSellRed, filled = true)
    }
}

private fun DrawScope.drawTriangle(center: Offset, size: Float, pointingUp: Boolean, color: Color, filled: Boolean) {
    val path = Path()
    if (pointingUp) {
        path.moveTo(center.x, center.y - size / 2)
        path.lineTo(center.x - size / 2, center.y + size / 2)
        path.lineTo(center.x + size / 2, center.y + size / 2)
    } else {
        path.moveTo(center.x, center.y + size / 2)
        path.lineTo(center.x - size / 2, center.y - size / 2)
        path.lineTo(center.x + size / 2, center.y - size / 2)
    }
    path.close()
    if (filled) drawPath(path, color)
    else        drawPath(path, color, style = Stroke(width = 2f))
}

private fun DrawScope.drawDiamond(center: Offset, size: Float, color: Color) {
    val path = Path().apply {
        moveTo(center.x, center.y - size / 2)
        lineTo(center.x + size / 2, center.y)
        lineTo(center.x, center.y + size / 2)
        lineTo(center.x - size / 2, center.y)
        close()
    }
    drawPath(path, color)
}

private fun priceToY(price: Float, priceMin: Float, priceRange: Float, chartHeight: Float, padding: Float): Float =
    chartPriceToY(price, priceMin, priceRange, chartHeight, padding)

@Composable
private fun ChartLegend(modifier: Modifier = Modifier) {
    val items = listOf(
        NovaBuyGreen      to "Regular BUY",
        NovaSellRed       to "Regular SELL",
        NovaExtendedBlue  to "Extended-hours",
        NovaGapPurple     to "Gap-driven",
        NovaFadedGray     to "Liquidity-filtered",
        NovaWarningYellow to "Macro-suppressed"
    )
    Column(modifier = modifier) {
        Text("Legend", style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
        Spacer(modifier = Modifier.height(4.dp))
        items.chunked(3).forEach { row ->
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                row.forEach { (color, label) ->
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 2.dp)) {
                        Canvas(modifier = Modifier.size(10.dp)) { drawCircle(color) }
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(label, style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        }
    }
}
