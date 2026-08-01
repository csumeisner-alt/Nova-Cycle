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
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
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
import com.novacycle.ui.components.ChartFreshnessHeader
import com.novacycle.ui.components.ChartPriceSummary
import com.novacycle.ui.components.ChartRenderMode
import com.novacycle.ui.components.CrosshairReadout
import com.novacycle.ui.components.PRICE_AXIS_WIDTH
import com.novacycle.ui.components.TIME_AXIS_HEIGHT
import com.novacycle.ui.components.candleIndexAt
import com.novacycle.ui.components.chartCrosshairInput
import com.novacycle.ui.components.chartPriceBounds
import com.novacycle.ui.components.chartPriceToY
import com.novacycle.ui.components.drawCloseLine
import com.novacycle.ui.components.drawCrosshair
import com.novacycle.ui.components.drawLastPriceMarker
import com.novacycle.ui.components.drawPriceAxis
import com.novacycle.ui.components.drawPriceReferenceLines
import com.novacycle.ui.components.drawTimeAxis
import com.novacycle.ui.components.signalIndexByCandle
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
    var renderMode by remember { mutableStateOf(ChartRenderMode.CANDLES) }
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
            Text("Raw Chart", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground)
            Text("VOO", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
        }

        // "Updated X ago" freshness label, ticking as time passes
        UpdatedAgoLabel(
            lastUpdatedAtMillis = uiState.lastUpdatedAtMillis,
            modifier = Modifier.padding(horizontal = 12.dp),
            extendedHoursAware = true
        )
        // Last candle timestamp + session, colored when stale
        ChartFreshnessHeader(
            lastCandle = uiState.candles.lastOrNull(),
            modifier = Modifier.padding(horizontal = 12.dp)
        )

        // Window + timeframe + render-mode selectors
        Row(modifier = Modifier.padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            windows.forEach { w ->
                FilterChip(
                    selected = uiState.selectedWindow == w,
                    onClick  = { viewModel.setWindow(w) },
                    label    = { Text(w) }
                )
            }
        }
        Row(modifier = Modifier.padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            TIMEFRAME_OPTIONS.forEach { (label, api) ->
                FilterChip(
                    selected = uiState.selectedTimeframe == api,
                    onClick  = { viewModel.setTimeframe(api) },
                    label    = { Text(label) }
                )
            }
            Spacer(Modifier.weight(1f))
            FilterChip(
                selected = renderMode == ChartRenderMode.LINE,
                onClick  = {
                    renderMode = if (renderMode == ChartRenderMode.LINE)
                        ChartRenderMode.CANDLES else ChartRenderMode.LINE
                },
                label = { Text(if (renderMode == ChartRenderMode.LINE) "Line" else "Candles") }
            )
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
                renderMode = renderMode,
                isIntraday = uiState.selectedTimeframe != "daily",
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

/** Chip label → API timeframe value shared by both chart screens. */
internal val TIMEFRAME_OPTIONS = listOf(
    "5m" to "5min",
    "15m" to "15min",
    "1h" to "1h",
    "1D" to "daily"
)

/**
 * Canvas candlestick chart with signal overlays, right price axis, bottom
 * time axis, last-price marker, line-mode rendering and a long-press
 * crosshair with OHLC readout.
 * Zoom: pinch via transformable modifier. Pan: drag via transformable.
 */
@Composable
fun CandlestickChart(
    candles: List<CandleResponse>,
    signals: List<SignalData>,
    priceSnapshot: com.novacycle.data.remote.models.PriceSnapshotResponse? = null,
    renderMode: ChartRenderMode = ChartRenderMode.CANDLES,
    isIntraday: Boolean = false,
    modifier: Modifier = Modifier,
    onSignalTapped: (SignalData) -> Unit = {}
) {
    if (candles.isEmpty()) return

    var scale   by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var crosshairIndex by remember(candles) { mutableStateOf<Int?>(null) }

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

    val axisLabelColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f)
    val gridColor      = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f)
    val crosshairColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
    val lineColor      = MaterialTheme.colorScheme.primary

    // Signals attach to the nearest candle bucket at or before their
    // timestamp, so markers survive timeframe changes.
    val signalIndex = remember(candles, signals) {
        signalIndexByCandle(candles, signals.map { it.timestamp })
    }
    val signalsByCandleIndex = remember(signalIndex, signals) {
        signals.mapNotNull { s -> signalIndex[s.timestamp]?.let { it to s } }.toMap()
    }

    val density = LocalDensity.current
    val axisWidthPx  = with(density) { PRICE_AXIS_WIDTH.toPx() }
    val axisHeightPx = with(density) { TIME_AXIS_HEIGHT.toPx() }
    val paddingPx    = with(density) { 8.dp.toPx() }

    var chartWidthPx by remember { mutableFloatStateOf(0f) }

    Box(
        modifier = modifier
            .onSizeChanged { chartWidthPx = it.width.toFloat() }
            .transformable(state = transformableState)
            .chartCrosshairInput(candles) { touchX, active ->
                crosshairIndex = if (!active || chartWidthPx <= 0f) null else {
                    val plotW = chartWidthPx - axisWidthPx
                    val barW = ((plotW / candles.size) * scale).coerceAtLeast(4f)
                    candleIndexAt(touchX, barW, offsetX, paddingPx, candles.size)
                }
            }
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val padding     = paddingPx
            val plotWidth   = size.width - axisWidthPx
            val plotHeight  = size.height - axisHeightPx
            val candleCount = candles.size
            val barWidth    = ((plotWidth / candleCount) * scale).coerceAtLeast(4f)
            val wickWidth   = (barWidth * 0.15f).coerceAtLeast(1f)

            // Axes and grid behind everything
            drawPriceAxis(priceMin, priceRange, plotWidth, plotHeight, padding,
                axisLabelColor, gridColor, textSizePx = 11.dp.toPx())
            drawTimeAxis(candles, barWidth, offsetX, padding, plotWidth, plotHeight,
                isIntraday, axisLabelColor, textSizePx = 10.dp.toPx())

            // Vertical session separators at every session-type transition
            // (pre-market → regular → after-hours), drawn behind the candles.
            drawSessionSeparators(candles, barWidth, offsetX, padding, plotHeight, plotWidth)
            priceSnapshot?.let {
                drawPriceReferenceLines(it, priceMin, priceRange, plotHeight, padding, plotWidth)
            }

            if (renderMode == ChartRenderMode.LINE) {
                drawCloseLine(candles, barWidth, offsetX, padding,
                    priceMin, priceRange, plotHeight, lineColor)
            }

            candles.forEachIndexed { index, candle ->
                val x = index * barWidth + offsetX + padding
                if (x + barWidth < 0 || x > plotWidth) return@forEachIndexed

                val highY  = priceToY(candle.high,  priceMin, priceRange, plotHeight, padding)
                val lowY   = priceToY(candle.low,   priceMin, priceRange, plotHeight, padding)

                if (renderMode == ChartRenderMode.CANDLES) {
                    val openY  = priceToY(candle.open,  priceMin, priceRange, plotHeight, padding)
                    val closeY = priceToY(candle.close, priceMin, priceRange, plotHeight, padding)

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
                }

                // Signal marker (drawn in both render modes)
                val signal = signalsByCandleIndex[index]
                if (signal != null) {
                    val markerSize = (barWidth * 0.8f).coerceIn(8f, 20f)
                    drawSignalMarker(signal, x + barWidth / 2, highY, lowY, markerSize)
                }
            }

            // Last price pinned to the axis on top of everything
            val lastPrice = priceSnapshot?.currentPrice ?: candles.last().close
            drawLastPriceMarker(lastPrice, priceMin, priceRange, plotWidth, plotHeight,
                padding, NovaWarningYellow, textSizePx = 10.dp.toPx())

            crosshairIndex?.let { idx ->
                candles.getOrNull(idx)?.let { c ->
                    drawCrosshair(idx, c, barWidth, offsetX, padding,
                        priceMin, priceRange, plotWidth, plotHeight, crosshairColor)
                }
            }
        }

        crosshairIndex?.let { idx ->
            candles.getOrNull(idx)?.let { c ->
                CrosshairReadout(candle = c, modifier = Modifier.align(Alignment.TopStart))
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
    padding: Float,
    plotHeight: Float = size.height,
    plotWidth: Float = size.width
) {
    val dash = PathEffect.dashPathEffect(floatArrayOf(8f, 8f))
    for (i in 1 until candles.size) {
        if (candles[i].sessionType != candles[i - 1].sessionType) {
            val x = i * barWidth + offsetX + padding
            if (x < 0 || x > plotWidth) continue
            drawLine(
                color = NovaExtendedBlue.copy(alpha = 0.35f),
                start = Offset(x, 0f),
                end   = Offset(x, plotHeight),
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

    // High-conviction signals get a subtle halo ring around the marker so
    // they stand out from ordinary opportunity-tier markers.
    if (signal.isHighConviction) {
        drawCircle(
            color = NovaWarningYellow.copy(alpha = 0.9f),
            radius = markerSize * 0.9f,
            center = Offset(centerX, cy),
            style = Stroke(width = 2.5f)
        )
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
                        Text(label, style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onBackground)
                    }
                }
            }
        }
    }
}
