package com.novacycle.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.novacycle.data.remote.models.ReliabilityMetricsResponse
import com.novacycle.data.remote.models.TradeCycleResponse
import com.novacycle.ui.components.PullRefreshBox
import com.novacycle.ui.components.UpdatedAgoLabel
import com.novacycle.ui.theme.NovaBuyGreen
import com.novacycle.ui.theme.NovaSellRed
import com.novacycle.viewmodel.ConfidenceBand
import com.novacycle.viewmodel.CycleSortColumn
import com.novacycle.viewmodel.PeriodFilter
import com.novacycle.viewmodel.ReliabilityUiState
import com.novacycle.viewmodel.ReliabilityViewModel
import com.novacycle.viewmodel.WinLossFilter
import java.util.Locale

/**
 * Reliability Metrics screen.
 *
 * Displays:
 *   • A summary panel (win rate, returns, hold time, best/worst)
 *   • Filter chips (date range, win/loss, volatility, liquidity, session type)
 *   • A sortable, horizontally-scrollable table of BUY→SELL cycles
 *   • Tap-to-expand rows with full cycle details
 *
 * The screen observes a single [ReliabilityUiState] from [ReliabilityViewModel].
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReliabilityScreen(
    onBack: () -> Unit = {},
    viewModel: ReliabilityViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Trade Reliability") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { viewModel.refresh() }) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh")
                    }
                }
            )
        }
    ) { padding ->
        PullRefreshBox(
            refreshing = uiState.isLoading,
            onRefresh = { viewModel.refresh() },
            contentIsScrollable = true,
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item { Spacer(Modifier.height(8.dp)) }

            item {
                // "Updated X ago" freshness label, ticking as time passes
                UpdatedAgoLabel(lastUpdatedAtMillis = uiState.lastUpdatedAtMillis, extendedHoursAware = true)
            }

            item {
                PeriodChipRow(
                    selected = uiState.periodFilter,
                    onSelected = { viewModel.setPeriodFilter(it) }
                )
            }

            item {
                ConfidenceChipRow(
                    selected = uiState.confidenceBand,
                    onSelected = { viewModel.setConfidenceBand(it) }
                )
            }

            item {
                SummaryPanel(
                    summary = uiState.summary,
                    isLoading = uiState.isLoading,
                    cycleCount = uiState.filteredCycles.size
                )
            }

            item {
                ModelPerformancePanel(uiState = uiState)
            }

            item {
                FilterPanel(
                    filters = uiState.filters,
                    onWinLossChanged = { viewModel.setWinLossFilter(it) },
                    onVolatilityChanged = { viewModel.setVolatilityClass(it) },
                    onLiquidityChanged = { viewModel.setLiquidityClass(it) },
                    onSessionChanged = { viewModel.setSessionType(it) },
                    onClear = { viewModel.clearFilters() }
                )
            }

            item {
                if (uiState.error != null) {
                    Text(
                        text = uiState.error ?: "",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }
            }

            if (uiState.filteredCycles.isEmpty()) {
                item {
                    if (uiState.isLoading) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 48.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "Loading cycles…",
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    } else {
                        EmptyCyclesCard()
                    }
                }
            } else {
                item {
                    CycleTableHeader(
                        sortColumn = uiState.sortColumn,
                        ascending = uiState.sortAscending,
                        onSort = { viewModel.setSortColumn(it) }
                    )
                }

                items(uiState.filteredCycles, key = { it.cycleId }) { cycle ->
                    CycleRow(
                        cycle = cycle,
                        isExpanded = uiState.expandedCycleId == cycle.cycleId,
                        onToggle = { viewModel.toggleExpanded(cycle.cycleId) }
                    )
                }
            }

            item { Spacer(Modifier.height(16.dp)) }
        }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary panel
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun SummaryPanel(
    summary: ReliabilityMetricsResponse?,
    isLoading: Boolean,
    cycleCount: Int
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(
                text = "Summary Metrics",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "$cycleCount filtered cycles",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(12.dp))

            if (isLoading && summary == null) {
                CircularProgressIndicator(Modifier.size(24.dp))
            } else if (summary == null) {
                Text("No data available", style = MaterialTheme.typography.bodyMedium)
            } else {
                val items = listOf(
                    "Win Rate" to "${(summary.winRate * 100).format1f()}%",
                    "Avg Return" to "${summary.averageReturnPercent.format2f()}%",
                    "Median Return" to "${summary.medianReturnPercent.format2f()}%",
                    "Avg Hold" to "${summary.averageHoldTime.format1f()} min",
                    "Best" to "${summary.bestTrade?.returnPercent?.format2f() ?: "--"}%",
                    "Worst" to "${summary.worstTrade?.returnPercent?.format2f() ?: "--"}%"
                )
                items.chunked(3).forEach { row ->
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        row.forEach { (label, value) ->
                            SummaryMetric(label, value, Modifier.weight(1f))
                        }
                        if (row.size < 3) {
                            repeat(3 - row.size) { Spacer(Modifier.weight(1f)) }
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                }
            }
        }
    }
}

@Composable
private fun SummaryMetric(label: String, value: String, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(value, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(label, style = MaterialTheme.typography.labelSmall, textAlign = TextAlign.Center)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Period + confidence chip rows
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun PeriodChipRow(
    selected: PeriodFilter,
    onSelected: (PeriodFilter) -> Unit
) {
    val options = listOf(
        "1D" to PeriodFilter.D1,
        "7D" to PeriodFilter.D7,
        "30D" to PeriodFilter.D30
    )
    Row(Modifier.padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("Period", style = MaterialTheme.typography.labelMedium, modifier = Modifier.width(72.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { (text, value) ->
                FilterChip(
                    selected = selected == value,
                    onClick = { onSelected(value) },
                    label = { Text(text) }
                )
            }
        }
    }
}

@Composable
private fun ConfidenceChipRow(
    selected: ConfidenceBand,
    onSelected: (ConfidenceBand) -> Unit
) {
    val options = listOf(
        "All" to ConfidenceBand.ALL,
        "Low" to ConfidenceBand.LOW,
        "Med" to ConfidenceBand.MEDIUM,
        "High" to ConfidenceBand.HIGH
    )
    Row(Modifier.padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("Confidence", style = MaterialTheme.typography.labelMedium, modifier = Modifier.width(72.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { (text, value) ->
                FilterChip(
                    selected = selected == value,
                    onClick = { onSelected(value) },
                    label = { Text(text) }
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model-performance panel
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun ModelPerformancePanel(uiState: ReliabilityUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(
                text = "Model Performance",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(12.dp))

            if (uiState.performanceError != null) {
                Text(
                    text = uiState.performanceError,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall
                )
                Spacer(Modifier.height(8.dp))
            }

            // Missed-rallies counter card alongside a cumulative P&L value.
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MissedRalliesCard(count = uiState.missedRallyCount, modifier = Modifier.weight(1f))
                CumulativePnlCard(percent = uiState.cumulativeReturnPercent, modifier = Modifier.weight(1f))
            }

            Spacer(Modifier.height(12.dp))

            // Calibration summary line under the win-rate area.
            val winRate = uiState.highConfidenceWinRate
            val claim = uiState.highConfidenceClaim
            val calibrationText = if (winRate != null) {
                "High-confidence calls win ${(winRate * 100).format1f()}% of the time " +
                    "(model claims ${(claim * 100).format1f()}%)"
            } else {
                "High-confidence calibration not available yet."
            }
            Text(
                text = calibrationText,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(Modifier.height(12.dp))

            RetrainAccuracyTrendSection(
                trend = uiState.accuracyTrend,
                latest = uiState.latestRetrainAccuracy,
                delta = uiState.retrainAccuracyDelta
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retrain accuracy trend
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compact per-retrain accuracy trend: a sparkline of out-of-sample accuracy
 * across retrains plus a "latest vs previous" delta summary. Entries with null
 * accuracy are already skipped upstream in [ReliabilityUiState.accuracyTrend].
 *
 * Tapping the card expands a list of individual retrains (date + accuracy %),
 * newest first.
 */
@Composable
private fun RetrainAccuracyTrendSection(
    trend: List<com.novacycle.data.remote.models.AccuracyHistoryEntry>,
    latest: Float?,
    delta: Float?
) {
    var expanded by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .then(
                if (latest != null) Modifier.clickable { expanded = !expanded } else Modifier
            ),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(Modifier.padding(12.dp)) {
            // Header row: title + expand/collapse chevron
            Row(
                Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Retrain Accuracy Trend",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold
                )
                if (latest != null) {
                    Icon(
                        imageVector = if (expanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.ArrowDropDown,
                        contentDescription = if (expanded) "Collapse retrain history" else "Expand retrain history",
                        modifier = Modifier.size(20.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            Spacer(Modifier.height(8.dp))

            if (latest == null) {
                Text(
                    text = "No retrain accuracy history yet.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                Row(
                    Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Column {
                        Text(
                            text = "${(latest * 100).format1f()}%",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Latest retrain",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    if (delta != null) {
                        val deltaColor = when {
                            delta > 0f -> NovaBuyGreen
                            delta < 0f -> NovaSellRed
                            else -> MaterialTheme.colorScheme.onSurfaceVariant
                        }
                        val prefix = if (delta >= 0f) "+" else "−"
                        Text(
                            text = "$prefix${(kotlin.math.abs(delta) * 100).format1f()} pts vs previous",
                            style = MaterialTheme.typography.bodySmall,
                            color = deltaColor,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                    Spacer(Modifier.weight(1f))
                    if (trend.size >= 2) {
                        AccuracySparkline(
                            values = trend.mapNotNull { it.accuracy },
                            modifier = Modifier
                                .width(120.dp)
                                .height(36.dp)
                        )
                    }
                }

                // Expanded per-retrain list, newest first
                if (expanded && trend.isNotEmpty()) {
                    Spacer(Modifier.height(12.dp))
                    Divider()
                    Spacer(Modifier.height(8.dp))
                    trend.asReversed().forEach { entry ->
                        RetrainHistoryRow(entry)
                        Spacer(Modifier.height(4.dp))
                    }
                }
            }
        }
    }
}

/**
 * A single row in the expanded retrain history: formatted date on the left,
 * accuracy percentage on the right. The [entry] is guaranteed to have a
 * non-null accuracy because [ReliabilityUiState.accuracyTrend] filters them out.
 */
@Composable
private fun RetrainHistoryRow(entry: com.novacycle.data.remote.models.AccuracyHistoryEntry) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f).padding(end = 8.dp)) {
            if (entry.modelName.isNotBlank()) {
                Text(
                    text = entry.modelName,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Text(
                text = formatIsoTimestamp(entry.trainedAt),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Text(
            text = "${((entry.accuracy ?: 0f) * 100).format1f()}%",
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.SemiBold
        )
    }
}

/**
 * Minimal line sparkline of accuracy values (0–1), scaled to the local
 * min/max range with a small pad so a flat trend renders mid-height.
 *
 * **Flat trend (rawRange ≤ 1e-6f)**
 * When all values are identical (or differ by < 1e-6f) the raw range would
 * cause division by zero. The guard substitutes range = 1f and shifts
 * effectiveMin down by 0.5 so the normalised position is 0.5 and the line
 * lands exactly at the vertical centre (pad + 0.5 * usable = height / 2),
 * not at the bottom edge.
 *
 * **Near-flat trend (rawRange just above 1e-6f)**
 * When the guard does NOT activate the real range is used for normalisation:
 *   yFor(v) = pad + (1 − (v − min) / rawRange) * usable
 * Because the formula divides by the actual spread, the minimum value always
 * maps to the bottom of the usable area (pad + usable) and the maximum always
 * maps to the top (pad), regardless of how small rawRange is.  The visual span
 * therefore equals the full usable height even when two retrains differ by only
 * a fraction of a percent — the sparkline is always visible and never a sliver.
 * No additional minimum-span clamp is needed.
 */
@Composable
private fun AccuracySparkline(values: List<Float>, modifier: Modifier = Modifier) {
    val lineColor = MaterialTheme.colorScheme.primary
    androidx.compose.foundation.Canvas(modifier = modifier) {
        if (values.size < 2) return@Canvas
        val min = values.min()
        val max = values.max()
        val rawRange = max - min
        val range = rawRange.takeIf { it > 1e-6f } ?: 1f
        // When the trend is flat, shift effectiveMin so normalised position = 0.5
        // and the line lands at mid-height instead of the bottom edge.
        val effectiveMin = if (rawRange > 1e-6f) min else min - 0.5f
        val stepX = size.width / (values.size - 1)
        val pad = size.height * 0.1f
        val usable = size.height - 2 * pad
        fun yFor(v: Float) = pad + (1f - (v - effectiveMin) / range) * usable
        for (i in 0 until values.size - 1) {
            drawLine(
                color = lineColor,
                start = androidx.compose.ui.geometry.Offset(i * stepX, yFor(values[i])),
                end = androidx.compose.ui.geometry.Offset((i + 1) * stepX, yFor(values[i + 1])),
                strokeWidth = 2.dp.toPx()
            )
        }
    }
}

@Composable
private fun MissedRalliesCard(count: Int, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = count.toString(),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Missed Rallies",
                style = MaterialTheme.typography.labelSmall,
                textAlign = TextAlign.Center
            )
        }
    }
}

@Composable
private fun CumulativePnlCard(percent: Float, modifier: Modifier = Modifier) {
    val color = when {
        percent > 0f -> NovaBuyGreen
        percent < 0f -> NovaSellRed
        else -> MaterialTheme.colorScheme.onSurface
    }
    val prefix = if (percent >= 0f) "+" else "−"
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "$prefix${kotlin.math.abs(percent).format2f()}%",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(
                text = "Cumulative P&L",
                style = MaterialTheme.typography.labelSmall,
                textAlign = TextAlign.Center
            )
        }
    }
}

@Composable
private fun EmptyCyclesCard() {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 24.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "No completed trades yet",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "Once BUY→SELL cycles complete for the selected period and " +
                    "confidence band, they'll appear here with full reliability metrics.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter panel
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun FilterPanel(
    filters: com.novacycle.viewmodel.CycleFilters,
    onWinLossChanged: (WinLossFilter) -> Unit,
    onVolatilityChanged: (String?) -> Unit,
    onLiquidityChanged: (String?) -> Unit,
    onSessionChanged: (String?) -> Unit,
    onClear: () -> Unit
) {
    Column {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Filters", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
            TextButton(onClick = onClear) {
                Icon(Icons.Filled.Clear, contentDescription = null, Modifier.size(18.dp))
                Spacer(Modifier.width(4.dp))
                Text("Clear")
            }
        }

        // Win/Loss chips
        FilterChipGroup(
            label = "Result",
            options = listOf("All" to WinLossFilter.ALL, "Win" to WinLossFilter.WIN, "Loss" to WinLossFilter.LOSS),
            selected = filters.winLoss,
            onSelected = onWinLossChanged
        )

        FilterOptionChipGroup(
            label = "Volatility",
            options = listOf(null, "low", "medium", "high"),
            selected = filters.volatilityClass,
            onSelected = onVolatilityChanged
        )

        FilterOptionChipGroup(
            label = "Liquidity",
            options = listOf(null, "adequate", "thin"),
            selected = filters.liquidityClass,
            onSelected = onLiquidityChanged
        )

        FilterOptionChipGroup(
            label = "Session",
            options = listOf(null, "pre_market", "regular", "after_hours"),
            selected = filters.sessionType,
            onSelected = onSessionChanged
        )
    }
}

@Composable
private fun <T> FilterChipGroup(
    label: String,
    options: List<Pair<String, T>>,
    selected: T,
    onSelected: (T) -> Unit
) {
    Row(Modifier.padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.labelMedium, modifier = Modifier.width(72.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { (text, value) ->
                FilterChip(
                    selected = selected == value,
                    onClick = { onSelected(value) },
                    label = { Text(text) }
                )
            }
        }
    }
}

@Composable
private fun FilterOptionChipGroup(
    label: String,
    options: List<String?>,
    selected: String?,
    onSelected: (String?) -> Unit
) {
    Row(Modifier.padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.labelMedium, modifier = Modifier.width(72.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { value ->
                val text = value ?: "All"
                FilterChip(
                    selected = selected == value,
                    onClick = { onSelected(value) },
                    label = { Text(text.replaceFirstChar { it.uppercase() }) }
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Table header
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun CycleTableHeader(
    sortColumn: CycleSortColumn,
    ascending: Boolean,
    onSort: (CycleSortColumn) -> Unit
) {
    val headerColor = MaterialTheme.colorScheme.surfaceVariant
    Row(
        Modifier
            .fillMaxWidth()
            .background(headerColor)
            .padding(vertical = 8.dp, horizontal = 12.dp)
            .horizontalScroll(rememberScrollState()),
        verticalAlignment = Alignment.CenterVertically
    ) {
        SortHeader("Return %", CycleSortColumn.RETURN_PERCENT, sortColumn, ascending, onSort, 80.dp)
        SortHeader("Return $", CycleSortColumn.RETURN_DOLLARS, sortColumn, ascending, onSort, 80.dp)
        SortHeader("Hold", CycleSortColumn.HOLD_TIME, sortColumn, ascending, onSort, 70.dp)
        SortHeader("Confidence", CycleSortColumn.CONFIDENCE, sortColumn, ascending, onSort, 90.dp)
        SortHeader("Gap", CycleSortColumn.GAP_TYPE, sortColumn, ascending, onSort, 70.dp)
        SortHeader("Liq", CycleSortColumn.LIQUIDITY_SCORE, sortColumn, ascending, onSort, 70.dp)
    }
}

@Composable
private fun RowScope.SortHeader(
    label: String,
    column: CycleSortColumn,
    currentColumn: CycleSortColumn,
    ascending: Boolean,
    onSort: (CycleSortColumn) -> Unit,
    width: androidx.compose.ui.unit.Dp
) {
    val selected = currentColumn == column
    Row(
        modifier = Modifier
            .width(width)
            .clickable { onSort(column) }
            .padding(end = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Start
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
        )
        if (selected) {
            Icon(
                imageVector = if (ascending) Icons.Filled.KeyboardArrowUp else Icons.Filled.ArrowDropDown,
                contentDescription = null,
                modifier = Modifier.size(16.dp)
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cycle row
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun CycleRow(
    cycle: TradeCycleResponse,
    isExpanded: Boolean,
    onToggle: () -> Unit
) {
    val returnColor = when {
        (cycle.returnPercent ?: 0f) > 0f -> NovaBuyGreen
        (cycle.returnPercent ?: 0f) < 0f -> NovaSellRed
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onToggle() },
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(Modifier.padding(12.dp)) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                verticalAlignment = Alignment.CenterVertically
            ) {
                TableCell(text = "${cycle.returnPercent?.format2f() ?: "--"}%", width = 80.dp, color = returnColor)
                TableCell(text = "${cycle.returnDollars?.format2f() ?: "--"}", width = 80.dp)
                TableCell(text = "${cycle.holdTimeMinutes?.format1f() ?: "--"}m", width = 70.dp)
                TableCell(text = "${cycle.confidenceAtBuy?.format1f() ?: "--"}", width = 90.dp)
                TableCell(text = cycle.gapTypeAtBuy ?: "--", width = 70.dp)
                TableCell(text = "${cycle.liquidityScoreAtBuy?.format2f() ?: "--"}", width = 70.dp)
            }

            if (isExpanded) {
                Spacer(Modifier.height(8.dp))
                Divider()
                Spacer(Modifier.height(8.dp))
                CycleDetail(cycle)
            }
        }
    }
}

@Composable
private fun RowScope.TableCell(
    text: String,
    width: androidx.compose.ui.unit.Dp,
    color: androidx.compose.ui.graphics.Color = MaterialTheme.colorScheme.onSurface
) {
    Text(
        text = text,
        color = color,
        style = MaterialTheme.typography.bodySmall,
        modifier = Modifier.width(width)
    )
}

@Composable
private fun CycleDetail(cycle: TradeCycleResponse) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        DetailLine("Cycle ID", cycle.cycleId)
        DetailLine("Buy", formatIsoTimestamp(cycle.buyTimestamp))
        DetailLine("Sell", formatIsoTimestamp(cycle.sellTimestamp))
        DetailLine("Buy Price", "${cycle.buyPrice?.format2f() ?: "--"}")
        DetailLine("Sell Price", "${cycle.sellPrice?.format2f() ?: "--"}")
        DetailLine("Confidence (sell)", "${cycle.confidenceAtSell?.format1f() ?: "--"}")
        DetailLine("Session", cycle.sessionTypeAtBuy ?: "--")
        DetailLine("Liquidity class", cycle.liquidityClass ?: "--")
        DetailLine("Volatility class", cycle.volatilityClass ?: "--")
        DetailLine("Macro override", if (cycle.macroOverrideApplied) "Yes" else "No")
    }
}

@Composable
private fun DetailLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth()) {
        Text(
            text = "$label: ",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.width(110.dp)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

private fun Float.format1f(): String = String.format(Locale.US, "%.1f", this)
private fun Float.format2f(): String = String.format(Locale.US, "%.2f", this)

// displayFormatter and formatIsoTimestamp live in ReliabilityScreenFormatters.kt
// so they can be called from unit tests with a pinned ZoneId.
