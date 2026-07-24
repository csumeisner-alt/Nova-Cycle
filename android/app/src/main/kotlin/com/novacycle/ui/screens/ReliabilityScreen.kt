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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.novacycle.data.remote.models.ReliabilityMetricsResponse
import com.novacycle.data.remote.models.TradeCycleResponse
import com.novacycle.ui.theme.NovaBuyGreen
import com.novacycle.ui.theme.NovaSellRed
import com.novacycle.viewmodel.CycleSortColumn
import com.novacycle.viewmodel.ReliabilityUiState
import com.novacycle.viewmodel.ReliabilityViewModel
import com.novacycle.viewmodel.WinLossFilter
import java.time.Instant
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
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
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item { Spacer(Modifier.height(8.dp)) }

            item {
                SummaryPanel(
                    summary = uiState.summary,
                    isLoading = uiState.isLoading,
                    cycleCount = uiState.filteredCycles.size
                )
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
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 48.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = if (uiState.isLoading) "Loading cycles…" else "No cycles match the current filters.",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
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

private val displayFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")

private fun formatIsoTimestamp(iso: String?): String {
    if (iso == null) return "--"
    return try {
        // Try offset/zoned ISO first (e.g. 2026-07-14T01:06:47Z)
        val instant = Instant.parse(iso)
        LocalDateTime.ofInstant(instant, ZoneId.systemDefault()).format(displayFormatter)
    } catch (e: Exception) {
        try {
            // Fall back to naive local ISO (e.g. 2026-07-14T01:06:47.323043)
            LocalDateTime.parse(iso, DateTimeFormatter.ISO_DATE_TIME).format(displayFormatter)
        } catch (e2: Exception) {
            iso
        }
    }
}
