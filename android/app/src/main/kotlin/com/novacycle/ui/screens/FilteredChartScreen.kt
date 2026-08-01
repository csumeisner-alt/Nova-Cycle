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
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.usecase.ApplyFilteredSignalsUseCase
import com.novacycle.ui.components.ConfidenceRibbon
import com.novacycle.ui.components.ChartFreshnessHeader
import com.novacycle.ui.components.ChartPriceSummary
import com.novacycle.ui.components.ChartRenderMode
import com.novacycle.ui.components.CrosshairReadout
import com.novacycle.ui.components.PRICE_AXIS_WIDTH
import com.novacycle.ui.components.PullRefreshBox
import com.novacycle.ui.components.SignalStoryCard
import com.novacycle.ui.components.TIME_AXIS_HEIGHT
import com.novacycle.ui.components.UpdatedAgoLabel
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
import com.novacycle.viewmodel.FilteredChartViewModel
import com.novacycle.viewmodel.SettingsViewModel
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Filtered Chart screen — strongest-confidence signals with trade-cycle shading and glow markers.
 */
@Composable
fun FilteredChartScreen(
    viewModel: FilteredChartViewModel = hiltViewModel(),
    settingsViewModel: SettingsViewModel = hiltViewModel()
) {
    val uiState  by viewModel.uiState.collectAsStateWithLifecycle()
    val settings by settingsViewModel.settings.collectAsStateWithLifecycle()

    LaunchedEffect(settings) { viewModel.applySettings(settings) }

    var selectedSignal by remember { mutableStateOf<SignalData?>(null) }
    val renderMode = if (uiState.renderMode == ChartRenderMode.LINE.name)
        ChartRenderMode.LINE else ChartRenderMode.CANDLES
    val windows = listOf("7d", "30d", "90d")

    PullRefreshBox(
        refreshing = uiState.isLoading,
        onRefresh = { viewModel.loadData() },
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
    ) {
    Column(modifier = Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(12.dp), Arrangement.SpaceBetween, Alignment.CenterVertically) {
            Text("Filtered Signals", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground)
            Text("VOO", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
        }

        // "Updated X ago" freshness label, ticking as time passes
        UpdatedAgoLabel(lastUpdatedAtMillis = uiState.lastUpdatedAtMillis, modifier = Modifier.padding(horizontal = 12.dp), extendedHoursAware = true)
        // Last candle timestamp + session, colored when stale
        ChartFreshnessHeader(lastCandle = uiState.candles.lastOrNull(), modifier = Modifier.padding(horizontal = 12.dp))

        Row(Modifier.padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            windows.forEach { w -> FilterChip(selected = uiState.selectedWindow == w, onClick = { viewModel.setWindow(w) }, label = { Text(w) }) }
        }
        Row(Modifier.padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
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
                    viewModel.setRenderMode(
                        if (renderMode == ChartRenderMode.LINE)
                            ChartRenderMode.CANDLES.name else ChartRenderMode.LINE.name
                    )
                },
                label = { Text(if (renderMode == ChartRenderMode.LINE) "Line" else "Candles") }
            )
        }

        if (uiState.isLoading) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        if (uiState.error != null) Text("⚠️ ${uiState.error}", color = NovaSellRed, modifier = Modifier.padding(12.dp))

        ChartPriceSummary(
            snapshot = uiState.priceSnapshot,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
        )

        if (uiState.filteredSignals.isNotEmpty()) {
            Row(Modifier.padding(horizontal = 12.dp, vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("▲ ${uiState.filteredSignals.count { it.isBuy }} BUY", color = NovaBuyGreen, style = MaterialTheme.typography.bodyMedium)
                Text("▼ ${uiState.filteredSignals.count { it.isSell }} SELL", color = NovaSellRed, style = MaterialTheme.typography.bodyMedium)
                Text("${uiState.tradeCycles.size} cycles", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onBackground)
            }
        }

        if (uiState.candles.isNotEmpty()) {
            FilteredCandlestickChart(
                candles = uiState.candles, signals = uiState.filteredSignals, cycles = uiState.tradeCycles,
                priceSnapshot = uiState.priceSnapshot,
                renderMode = renderMode,
                isIntraday = uiState.selectedTimeframe != "daily",
                modifier = Modifier.fillMaxWidth().weight(1f),
                onSignalTapped = { selectedSignal = it }
            )
        } else if (!uiState.isLoading) {
            Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) { Text("No chart data", color = NovaNeutralGray) }
        }

        val momentumProxy = uiState.filteredSignals.mapIndexed { i, s ->
            if (i == 0) 0f else s.confidence - uiState.filteredSignals[i - 1].confidence
        }
        ConfidenceRibbon(momentumPoints = momentumProxy, modifier = Modifier.padding(12.dp))
    }
    }

    selectedSignal?.let { signal ->
        SignalStoryCard(signal = signal, storyLevel = settings.storyCardLevel, onDismiss = { selectedSignal = null })
    }
}

@Composable
private fun FilteredCandlestickChart(
    candles: List<CandleResponse>,
    signals: List<SignalData>,
    cycles: List<ApplyFilteredSignalsUseCase.TradeCycle>,
    priceSnapshot: com.novacycle.data.remote.models.PriceSnapshotResponse? = null,
    renderMode: ChartRenderMode = ChartRenderMode.CANDLES,
    isIntraday: Boolean = false,
    modifier: Modifier = Modifier,
    onSignalTapped: (SignalData) -> Unit = {}
) {
    var scale   by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var crosshairIndex by remember(candles) { mutableStateOf<Int?>(null) }
    var chartWidthPx by remember { mutableFloatStateOf(0f) }
    val transformableState = rememberTransformableState { zoom, pan, _ ->
        scale   = (scale * zoom).coerceIn(0.5f, 8f)
        offsetX = (offsetX + pan.x).coerceIn(-8000f, 0f)
    }

    val (priceMin, boundedPriceMax) = chartPriceBounds(
        candles.minOf { it.low },
        candles.maxOf { it.high },
        priceSnapshot
    )
    val priceMax   = boundedPriceMax
    val priceRange = (priceMax - priceMin).coerceAtLeast(0.01f)

    // Signals and cycle boundaries attach to the nearest candle bucket at or
    // before their timestamps, so overlays survive timeframe changes.
    val allTimestamps = remember(candles, signals, cycles) {
        (signals.map { it.timestamp } +
            cycles.flatMap { listOfNotNull(it.buySignal.timestamp, it.sellSignal?.timestamp) })
            .distinct()
    }
    val tsToIdx = remember(candles, allTimestamps) { signalIndexByCandle(candles, allTimestamps) }
    val sigByCandleIdx = remember(tsToIdx, signals) {
        signals.mapNotNull { s -> tsToIdx[s.timestamp]?.let { it to s } }.toMap()
    }

    val axisLabelColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f)
    val gridColor      = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f)
    val crosshairColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
    val lineColor      = MaterialTheme.colorScheme.primary

    val density = LocalDensity.current
    val axisWidthPx  = with(density) { PRICE_AXIS_WIDTH.toPx() }
    val axisHeightPx = with(density) { TIME_AXIS_HEIGHT.toPx() }
    val paddingPx    = with(density) { 8.dp.toPx() }

    Box(
        modifier = modifier
            .onSizeChanged { chartWidthPx = it.width.toFloat() }
            .transformable(transformableState)
            .chartCrosshairInput(candles) { touchX, active ->
                crosshairIndex = if (!active || chartWidthPx <= 0f) null else {
                    val plotW = chartWidthPx - axisWidthPx
                    val barW = ((plotW / candles.size) * scale).coerceAtLeast(4f)
                    candleIndexAt(touchX, barW, offsetX, paddingPx, candles.size)
                }
            }
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val padding    = paddingPx
            val plotWidth  = size.width - axisWidthPx
            val plotHeight = size.height - axisHeightPx
            val barWidth   = ((plotWidth / candles.size) * scale).coerceAtLeast(4f)
            val wickW      = (barWidth * 0.15f).coerceAtLeast(1f)

            // Axes and grid behind everything
            drawPriceAxis(priceMin, priceRange, plotWidth, plotHeight, padding,
                axisLabelColor, gridColor, textSizePx = 11.dp.toPx())
            drawTimeAxis(candles, barWidth, offsetX, padding, plotWidth, plotHeight,
                isIntraday, axisLabelColor, textSizePx = 10.dp.toPx())

            // Vertical session separators (shared with RawChartScreen)
            drawSessionSeparators(candles, barWidth, offsetX, padding, plotHeight, plotWidth)
            priceSnapshot?.let {
                drawPriceReferenceLines(it, priceMin, priceRange, plotHeight, padding, plotWidth)
            }

            // Index ranges covered by completed BUY→SELL cycles, so candles
            // inside a cycle can carry a subtle trend tint.
            val cycleRanges = cycles.mapNotNull { cycle ->
                val bi = tsToIdx[cycle.buySignal.timestamp] ?: return@mapNotNull null
                val si = cycle.sellSignal?.let { tsToIdx[it.timestamp] } ?: return@mapNotNull null
                bi..si
            }

            // Trade-cycle shading
            cycles.forEach { cycle ->
                val bi = tsToIdx[cycle.buySignal.timestamp] ?: return@forEach
                val si = cycle.sellSignal?.let { tsToIdx[it.timestamp] } ?: return@forEach
                val sx = bi * barWidth + offsetX + padding
                val ex = si * barWidth + offsetX + padding + barWidth
                if (ex > 0 && sx < plotWidth) {
                    drawRect(NovaBuyGreen.copy(alpha = 0.07f),
                        Offset(max(sx, 0f), 0f), Size(min(ex, plotWidth) - max(sx, 0f), plotHeight))
                }
            }

            if (renderMode == ChartRenderMode.LINE) {
                drawCloseLine(candles, barWidth, offsetX, padding,
                    priceMin, priceRange, plotHeight, lineColor)
            }

            // Candles + signal markers
            candles.forEachIndexed { idx, candle ->
                val x = idx * barWidth + offsetX + padding
                if (x + barWidth < 0 || x > plotWidth) return@forEachIndexed

                val highY  = priceToY(candle.high,  priceMin, priceRange, plotHeight, padding)
                val lowY   = priceToY(candle.low,   priceMin, priceRange, plotHeight, padding)

                if (renderMode == ChartRenderMode.CANDLES) {
                    val openY  = priceToY(candle.open,  priceMin, priceRange, plotHeight, padding)
                    val closeY = priceToY(candle.close, priceMin, priceRange, plotHeight, padding)
                    // Subtle trend tint: candles inside a BUY cycle lean slightly
                    // toward green — bullish/bearish base semantics stay intact.
                    val inBuyCycle = cycleRanges.any { idx in it }
                    val baseColor  = if (candle.close >= candle.open) NovaBuyGreen else NovaSellRed
                    val color = if (inBuyCycle) {
                        lerp(baseColor, NovaBuyGreen, 0.18f).copy(alpha = 0.85f)
                    } else baseColor.copy(alpha = 0.8f)

                    drawLine(color, Offset(x + barWidth/2, highY), Offset(x + barWidth/2, lowY), wickW)
                    drawRect(color, Offset(x + wickW, minOf(openY, closeY)),
                        Size((barWidth - wickW*2).coerceAtLeast(1f), abs(openY - closeY).coerceAtLeast(1f)))
                }

                // Signal markers (drawn in both render modes)
                sigByCandleIdx[idx]?.let { signal ->
                    val ms = (barWidth * 1.2f).coerceIn(12f, 28f)
                    val cx = x + barWidth / 2
                    val cy = if (signal.isBuy) lowY + ms*0.6f + ms else highY - ms*0.6f - ms
                    val mc = if (signal.isBuy) NovaBuyGreen else NovaSellRed
                    drawFilledTriangle(signal.isBuy, cx, cy, ms * 1.8f, mc.copy(alpha = 0.2f))
                    drawFilledTriangle(signal.isBuy, cx, cy, ms, mc)
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

private fun DrawScope.drawFilledTriangle(isBuy: Boolean, cx: Float, cy: Float, size: Float, color: Color) {
    val path = Path()
    if (isBuy) { path.moveTo(cx, cy - size/2); path.lineTo(cx - size/2, cy + size/2); path.lineTo(cx + size/2, cy + size/2) }
    else       { path.moveTo(cx, cy + size/2); path.lineTo(cx - size/2, cy - size/2); path.lineTo(cx + size/2, cy - size/2) }
    path.close()
    drawPath(path, color)
}

private fun priceToY(price: Float, priceMin: Float, priceRange: Float, h: Float, padding: Float): Float =
    chartPriceToY(price, priceMin, priceRange, h, padding)
