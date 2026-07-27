package com.novacycle.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.domain.model.ConfidencePoint
import com.novacycle.domain.model.SmoothingMode
import com.novacycle.ui.components.PullRefreshBox
import com.novacycle.ui.components.UpdatedAgoLabel
import com.novacycle.ui.theme.*
import com.novacycle.viewmodel.ConfidenceHistoryViewModel
import com.novacycle.viewmodel.SettingsViewModel

/**
 * Confidence History screen: dual line chart (Long BUY = green, Short BUY = blue).
 * Extended-hours segments rendered at reduced opacity.
 * EMA toggle persists via SensitivitySettings.
 */
@Composable
fun ConfidenceHistoryScreen(
    viewModel: ConfidenceHistoryViewModel = hiltViewModel(),
    settingsViewModel: SettingsViewModel = hiltViewModel()
) {
    val uiState  by viewModel.uiState.collectAsStateWithLifecycle()
    val settings by settingsViewModel.settings.collectAsStateWithLifecycle()

    LaunchedEffect(settings) { viewModel.applySettings(settings) }

    val windows = listOf("24h", "7d", "30d")

    PullRefreshBox(
        refreshing = uiState.isLoading,
        onRefresh = { viewModel.loadHistory() },
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
    ) {
    Column(modifier = Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(12.dp), Arrangement.SpaceBetween, Alignment.CenterVertically) {
            Text("Confidence History", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("EMA", style = MaterialTheme.typography.labelSmall)
                Spacer(Modifier.width(4.dp))
                Switch(
                    checked = settings.smoothingMode != SmoothingMode.RAW,
                    onCheckedChange = { settingsViewModel.updateSmoothingMode(if (it) SmoothingMode.EMA else SmoothingMode.RAW) }
                )
            }
        }

        // "Updated X ago" freshness label, ticking as time passes
        UpdatedAgoLabel(lastUpdatedAtMillis = uiState.lastUpdatedAtMillis, modifier = Modifier.padding(horizontal = 12.dp), extendedHoursAware = true)

        Row(Modifier.padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            windows.forEach { w -> FilterChip(selected = uiState.selectedWindow == w, onClick = { viewModel.setWindow(w) }, label = { Text(w) }) }
        }

        if (uiState.isLoading) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        if (uiState.error != null) Text("⚠️ ${uiState.error}", color = NovaSellRed, modifier = Modifier.padding(12.dp))

        if (uiState.confidencePoints.isNotEmpty()) {
            ConfidenceLineChart(uiState.confidencePoints, Modifier.fillMaxWidth().weight(1f).padding(12.dp))
        } else if (!uiState.isLoading) {
            Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
                Text("No confidence data available", color = NovaNeutralGray)
            }
        }

        ConfidenceLegend(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp))
    }
    }
}

@Composable
private fun ConfidenceLineChart(points: List<ConfidencePoint>, modifier: Modifier = Modifier) {
    if (points.size < 2) return
    Canvas(modifier = modifier) {
        val count   = points.size
        val stepX   = size.width / (count - 1).coerceAtLeast(1)
        val padding = 8.dp.toPx()
        val usableH = size.height - padding * 2

        fun confToY(v: Float) = size.height - padding - (v / 100f).coerceIn(0f, 1f) * usableH

        listOf(25f, 50f, 75f).forEach { level ->
            drawLine(Color.Gray.copy(alpha = 0.2f), Offset(0f, confToY(level)), Offset(size.width, confToY(level)), 1f)
        }

        // Gradient fill under each confidence line (line color fading to
        // transparent toward the chart floor) for a premium filled-area look.
        fun fillUnder(values: List<Float>, color: Color, topAlpha: Float) {
            val path = Path().apply {
                moveTo(0f, size.height - padding)
                values.forEachIndexed { i, v -> lineTo(i * stepX, confToY(v)) }
                lineTo((count - 1) * stepX, size.height - padding)
                close()
            }
            drawPath(
                path,
                brush = Brush.verticalGradient(
                    colors = listOf(color.copy(alpha = topAlpha), color.copy(alpha = 0f)),
                    startY = padding,
                    endY   = size.height - padding
                )
            )
        }
        fillUnder(points.map { it.shortBuyConfidence }, NovaExtendedBlue, 0.12f)
        fillUnder(points.map { it.longBuyConfidence }, NovaBuyGreen, 0.18f)

        for (i in 1 until count) {
            val prev  = points[i - 1]
            val curr  = points[i]
            val alpha = if (curr.isExtendedHours) 0.35f else 0.9f
            drawLine(NovaBuyGreen.copy(alpha = alpha),
                Offset((i-1)*stepX, confToY(prev.longBuyConfidence)), Offset(i*stepX, confToY(curr.longBuyConfidence)),
                strokeWidth = 2.5f, cap = StrokeCap.Round)
            drawLine(NovaExtendedBlue.copy(alpha = alpha),
                Offset((i-1)*stepX, confToY(prev.shortBuyConfidence)), Offset(i*stepX, confToY(curr.shortBuyConfidence)),
                strokeWidth = 2.5f, cap = StrokeCap.Round)
        }

        for (i in 1 until count) {
            val prev = points[i-1].longBuyConfidence
            val curr = points[i].longBuyConfidence
            if ((prev < 50f && curr >= 50f) || (prev >= 50f && curr < 50f)) {
                drawLine(NovaWarningYellow.copy(alpha = 0.5f), Offset(i*stepX, 0f), Offset(i*stepX, size.height), 1.5f)
            }
        }
    }
}

@Composable
private fun ConfidenceLegend(modifier: Modifier = Modifier) {
    Row(modifier = modifier, horizontalArrangement = Arrangement.spacedBy(20.dp)) {
        LegendLine(NovaBuyGreen, "Long BUY")
        LegendLine(NovaExtendedBlue, "Short BUY")
        LegendLine(NovaBuyGreen.copy(alpha = 0.35f), "Extended-hrs")
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
