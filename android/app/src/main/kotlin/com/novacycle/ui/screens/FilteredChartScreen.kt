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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.usecase.ApplyFilteredSignalsUseCase
import com.novacycle.ui.components.ConfidenceRibbon
import com.novacycle.ui.components.PullRefreshBox
import com.novacycle.ui.components.SignalStoryCard
import com.novacycle.ui.components.UpdatedAgoLabel
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
    val windows = listOf("7d", "30d", "90d")

    PullRefreshBox(
        refreshing = uiState.isLoading,
        onRefresh = { viewModel.loadData() },
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
    ) {
    Column(modifier = Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(12.dp), Arrangement.SpaceBetween, Alignment.CenterVertically) {
            Text("Filtered Signals", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Text("VOO", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
        }

        // "Updated X ago" freshness label, ticking as time passes
        UpdatedAgoLabel(lastUpdatedAtMillis = uiState.lastUpdatedAtMillis, modifier = Modifier.padding(horizontal = 12.dp))

        Row(Modifier.padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            windows.forEach { w -> FilterChip(selected = uiState.selectedWindow == w, onClick = { viewModel.setWindow(w) }, label = { Text(w) }) }
        }

        if (uiState.isLoading) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        if (uiState.error != null) Text("⚠️ ${uiState.error}", color = NovaSellRed, modifier = Modifier.padding(12.dp))

        if (uiState.filteredSignals.isNotEmpty()) {
            Row(Modifier.padding(horizontal = 12.dp, vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("▲ ${uiState.filteredSignals.count { it.isBuy }} BUY", color = NovaBuyGreen, style = MaterialTheme.typography.bodyMedium)
                Text("▼ ${uiState.filteredSignals.count { it.isSell }} SELL", color = NovaSellRed, style = MaterialTheme.typography.bodyMedium)
                Text("${uiState.tradeCycles.size} cycles", style = MaterialTheme.typography.bodyMedium)
            }
        }

        if (uiState.candles.isNotEmpty()) {
            FilteredCandlestickChart(
                candles = uiState.candles, signals = uiState.filteredSignals, cycles = uiState.tradeCycles,
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
    modifier: Modifier = Modifier,
    onSignalTapped: (SignalData) -> Unit = {}
) {
    var scale   by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    val transformableState = rememberTransformableState { zoom, pan, _ ->
        scale   = (scale * zoom).coerceIn(0.5f, 8f)
        offsetX = (offsetX + pan.x).coerceIn(-8000f, 0f)
    }

    val priceMin   = candles.minOf { it.low }
    val priceMax   = candles.maxOf { it.high }
    val priceRange = (priceMax - priceMin).coerceAtLeast(0.01f)
    val tsToIdx    = candles.mapIndexed { i, c -> c.timestamp to i }.toMap()
    val sigByTs    = signals.associateBy { it.timestamp }

    Box(modifier = modifier.transformable(transformableState)) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val padding  = 8.dp.toPx()
            val barWidth = ((size.width / candles.size) * scale).coerceAtLeast(4f)
            val wickW    = (barWidth * 0.15f).coerceAtLeast(1f)

            // Trade-cycle shading
            cycles.forEach { cycle ->
                val bi = tsToIdx[cycle.buySignal.timestamp] ?: return@forEach
                val si = cycle.sellSignal?.let { tsToIdx[it.timestamp] } ?: return@forEach
                val sx = bi * barWidth + offsetX + padding
                val ex = si * barWidth + offsetX + padding + barWidth
                if (ex > 0 && sx < size.width) {
                    drawRect(NovaBuyGreen.copy(alpha = 0.07f),
                        Offset(max(sx, 0f), 0f), Size(min(ex, size.width) - max(sx, 0f), size.height))
                }
            }

            // Candles + signal markers
            candles.forEachIndexed { idx, candle ->
                val x = idx * barWidth + offsetX + padding
                if (x + barWidth < 0 || x > size.width) return@forEachIndexed

                val openY  = priceToY(candle.open,  priceMin, priceRange, size.height, padding)
                val closeY = priceToY(candle.close, priceMin, priceRange, size.height, padding)
                val highY  = priceToY(candle.high,  priceMin, priceRange, size.height, padding)
                val lowY   = priceToY(candle.low,   priceMin, priceRange, size.height, padding)
                val color  = if (candle.close >= candle.open) NovaBuyGreen.copy(alpha = 0.8f) else NovaSellRed.copy(alpha = 0.8f)

                drawLine(color, Offset(x + barWidth/2, highY), Offset(x + barWidth/2, lowY), wickW)
                drawRect(color, Offset(x + wickW, minOf(openY, closeY)),
                    Size((barWidth - wickW*2).coerceAtLeast(1f), abs(openY - closeY).coerceAtLeast(1f)))

                sigByTs[candle.timestamp]?.let { signal ->
                    val ms = (barWidth * 1.2f).coerceIn(12f, 28f)
                    val cx = x + barWidth / 2
                    val cy = if (signal.isBuy) lowY + ms*0.6f + ms else highY - ms*0.6f - ms
                    val mc = if (signal.isBuy) NovaBuyGreen else NovaSellRed
                    drawFilledTriangle(signal.isBuy, cx, cy, ms * 1.8f, mc.copy(alpha = 0.2f))
                    drawFilledTriangle(signal.isBuy, cx, cy, ms, mc)
                }
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
    h - padding - ((price - priceMin) / priceRange) * (h - padding * 2)
