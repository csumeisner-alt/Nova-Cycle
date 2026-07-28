package com.novacycle.ui.screens

import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.domain.model.ConfidencePoint
import com.novacycle.domain.model.SmoothingMode
import com.novacycle.ui.components.PullRefreshBox
import com.novacycle.ui.components.UpdatedAgoLabel
import com.novacycle.ui.components.confidence.ConfidenceChart
import com.novacycle.ui.components.confidence.ConfidenceLegend
import com.novacycle.ui.theme.*
import com.novacycle.viewmodel.ConfidenceHistoryViewModel
import com.novacycle.viewmodel.SettingsViewModel
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Confidence History screen: dual line chart of buy confidence per gauge —
 * Buy (Long-term) = green, Buy (Short-term) = blue.
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

    val windows = listOf("3h", "6h", "12h", "24h", "7d", "30d", "3mo", "6mo")
    var showEmaInfo by remember { mutableStateOf(false) }
    val emaEnabled = settings.smoothingMode != SmoothingMode.RAW

    PullRefreshBox(
        refreshing = uiState.isLoading,
        onRefresh = { viewModel.loadHistory() },
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
    ) {
    Column(modifier = Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(12.dp), Arrangement.SpaceBetween, Alignment.CenterVertically) {
            Text("Confidence History", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "EMA (Smooth Confidence)",
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(end = 2.dp)
                )
                TextButton(onClick = { showEmaInfo = true }, contentPadding = PaddingValues(0.dp),
                    modifier = Modifier.size(24.dp)) {
                    Text("ⓘ", style = MaterialTheme.typography.labelSmall)
                }
                Spacer(Modifier.width(2.dp))
                Switch(
                    checked = emaEnabled,
                    onCheckedChange = { settingsViewModel.updateSmoothingMode(if (it) SmoothingMode.EMA else SmoothingMode.RAW) }
                )
            }
        }

        if (showEmaInfo) {
            AlertDialog(
                onDismissRequest = { showEmaInfo = false },
                confirmButton = { TextButton(onClick = { showEmaInfo = false }) { Text("Got it") } },
                title = { Text("EMA (Smooth Confidence)") },
                text = {
                    Text(
                        "EMA (Exponential Moving Average) smooths the confidence lines by " +
                        "giving recent readings more weight than older ones. This reduces " +
                        "noise so trends are easier to see, at the cost of reacting slightly " +
                        "slower to sudden changes. Turn it off to see raw values."
                    )
                }
            )
        }

        // "Updated X ago" freshness label, ticking as time passes
        UpdatedAgoLabel(lastUpdatedAtMillis = uiState.lastUpdatedAtMillis, modifier = Modifier.padding(horizontal = 12.dp), extendedHoursAware = true)

        Row(
            Modifier.padding(horizontal = 12.dp).horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            windows.forEach { w -> FilterChip(selected = uiState.selectedWindow == w, onClick = { viewModel.setWindow(w) }, label = { Text(w) }) }
        }

        if (uiState.isLoading) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        if (uiState.error != null) Text("⚠️ ${uiState.error}", color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(12.dp))

        // Legend above the chart + trend mini-summary
        ConfidenceLegend(modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp))
        if (uiState.confidencePoints.size >= 2) {
            TrendSummary(
                points = uiState.confidencePoints,
                window = uiState.selectedWindow,
                modifier = Modifier.padding(horizontal = 12.dp)
            )
        }

        // Animated transition between datasets when the range changes.
        Crossfade(
            targetState = uiState.confidencePoints,
            animationSpec = tween(durationMillis = 350),
            modifier = Modifier.fillMaxWidth().weight(1f),
            label = "confidence-chart"
        ) { points ->
            if (points.size >= 2) {
                ConfidenceChart(
                    points = points,
                    windowLabel = uiState.selectedWindow,
                    emaEnabled = emaEnabled,
                    modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 4.dp)
                )
            } else if (!uiState.isLoading) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No data available for selected period", color = NovaNeutralGray)
                }
            }
        }
    }
    }
}

/**
 * Mini-summary above the chart: trend direction arrows and net change of each
 * series over the loaded window, e.g. "▲ Buy (Long-term) rising +4% over last 24h".
 */
@Composable
private fun TrendSummary(
    points: List<ConfidencePoint>,
    window: String,
    modifier: Modifier = Modifier
) {
    val longDelta = points.last().longBuyConfidence - points.first().longBuyConfidence
    val shortDelta = points.last().shortBuyConfidence - points.first().shortBuyConfidence
    Column(modifier = modifier) {
        SummaryLine("Buy (Long-term)", longDelta, window, NovaBuyGreen)
        SummaryLine("Buy (Short-term)", shortDelta, window, NovaExtendedBlue)
    }
}

@Composable
private fun SummaryLine(name: String, delta: Float, window: String, color: androidx.compose.ui.graphics.Color) {
    val rounded = delta.roundToInt()
    val (arrow, verb) = when {
        rounded > 0 -> "▲" to "rising"
        rounded < 0 -> "▼" to "weakening"
        else -> "▶" to "flat"
    }
    val change = if (rounded == 0) "" else " ${if (rounded > 0) "+" else "−"}${abs(rounded)}%"
    Text(
        "$arrow $name $verb$change over last $window",
        style = MaterialTheme.typography.labelSmall,
        color = color.copy(alpha = 0.9f)
    )
}
