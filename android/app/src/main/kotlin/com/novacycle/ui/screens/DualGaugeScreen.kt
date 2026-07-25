package com.novacycle.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.domain.model.GaugeState
import com.novacycle.ui.components.DualGaugeWidget
import com.novacycle.ui.components.TickerSelector
import com.novacycle.ui.theme.*
import com.novacycle.viewmodel.DualGaugeViewModel

/**
 * Main dashboard screen.
 * Shows two gauge widgets (Long-Trend + Short-Trend) side by side,
 * hold-time estimate, and a quick indicator summary row.
 * Pull-to-refresh triggers parallel reload of all data.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DualGaugeScreen(
    viewModel: DualGaugeViewModel = hiltViewModel(),
    onNavigateToRawChart: () -> Unit = {},
    onNavigateToHoldTime: () -> Unit = {},
    onNavigateToReliability: () -> Unit = {}
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val longGaugeState = GaugeState(
        score      = uiState.longPrediction?.score ?: 0f,
        signal     = uiState.longPrediction?.signal ?: "neutral",
        confidence = uiState.longPrediction?.confidence ?: 0f,
        gaugeType  = "long",
        ticker     = uiState.selectedTicker,
        isLoading  = uiState.isLoading && uiState.longPrediction == null
    )
    val shortGaugeState = GaugeState(
        score      = uiState.shortPrediction?.score ?: 0f,
        signal     = uiState.shortPrediction?.signal ?: "neutral",
        confidence = uiState.shortPrediction?.confidence ?: 0f,
        gaugeType  = "short",
        ticker     = uiState.selectedTicker,
        isLoading  = uiState.isLoading && uiState.shortPrediction == null
    )

    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // ── Header row ────────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "NovaCycle",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    TickerSelector(
                        selectedTicker   = uiState.selectedTicker,
                        onTickerSelected = { viewModel.selectTicker(it) }
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(onClick = { viewModel.refreshAll() }) {
                        Icon(
                            imageVector      = Icons.Filled.Refresh,
                            contentDescription = "Refresh",
                            tint             = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Error banner
            if (uiState.error != null) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = NovaSellRed.copy(alpha = 0.15f)
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text  = "⚠️ ${uiState.error}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = NovaSellRed,
                        modifier = Modifier.padding(12.dp)
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
            }

            // ── Backend-unreachable notice ───────────────────────────────
            // Shown after several consecutive failed /healthz polls; visually
            // distinct (red outline card) from the amber degraded banner.
            if (uiState.backendUnreachable) {
                OutlinedCard(
                    colors = CardDefaults.outlinedCardColors(
                        containerColor = NovaSellRed.copy(alpha = 0.08f)
                    ),
                    border = androidx.compose.foundation.BorderStroke(1.dp, NovaSellRed),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text       = "🔌 Backend unreachable",
                            style      = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color      = NovaSellRed
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text  = "Backend unreachable — data may be stale.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = NovaSellRed.copy(alpha = 0.9f)
                        )
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
            }

            // ── Degraded-predictions warning banner ──────────────────────
            // Mirrors the web status page: shown while /healthz reports
            // status "degraded", naming the affected model(s) and alerts.
            val health = uiState.health
            if (health?.isDegraded == true) {
                val amber = Color(0xFFFFB300)
                val degradedModels = health.degradedModels
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = amber.copy(alpha = 0.15f)
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text       = "⚠️ Predictions degraded",
                            style      = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color      = amber
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = if (degradedModels.isNotEmpty()) {
                                "Predictions may be unreliable — affected model" +
                                    (if (degradedModels.size > 1) "s" else "") +
                                    ": ${degradedModels.joinToString(", ")}."
                            } else {
                                "Some system components are degraded."
                            },
                            style = MaterialTheme.typography.bodyMedium,
                            color = amber.copy(alpha = 0.9f)
                        )
                        health.alerts.orEmpty().forEach { alert ->
                            Text(
                                text  = "• $alert",
                                style = MaterialTheme.typography.bodySmall,
                                color = amber.copy(alpha = 0.7f)
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
            }

            // ── Dual Gauge widgets ────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                DualGaugeWidget(
                    gaugeState = longGaugeState,
                    label      = "Long-Trend",
                    modifier   = Modifier.weight(1f)
                )
                DualGaugeWidget(
                    gaugeState = shortGaugeState,
                    label      = "Short-Trend",
                    modifier   = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── Hold-Time card ────────────────────────────────────────────
            val holdTime = uiState.holdTime
            Card(
                onClick  = onNavigateToHoldTime,
                modifier = Modifier.fillMaxWidth(),
                colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment     = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            "Hold Time Estimate",
                            style      = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text       = holdTime?.humanReadable ?: if (uiState.isLoading) "Loading…" else "Tap to load",
                            style      = MaterialTheme.typography.bodyLarge,
                            color      = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    if (holdTime != null) {
                        ConfidenceBadge(holdTime.confidence)
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── Indicator summary row ─────────────────────────────────────
            val indicators = uiState.indicators
            if (indicators != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            "Indicator Snapshot",
                            style      = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            IndicatorChip(
                                label = "RSI",
                                value = "%.1f".format(indicators.rsi),
                                color = when {
                                    indicators.rsi > 70 -> NovaSellRed
                                    indicators.rsi < 30 -> NovaBuyGreen
                                    else                -> NovaNeutralGray
                                }
                            )
                            IndicatorChip(
                                label = "ADX",
                                value = "%.1f".format(indicators.adx),
                                color = if (indicators.adx > 25) NovaBuyGreen else NovaNeutralGray
                            )
                            IndicatorChip(
                                label = "VIX",
                                value = indicators.vixRegime.uppercase(),
                                color = when (indicators.vixRegime.lowercase()) {
                                    "low"     -> VixLow
                                    "normal"  -> VixNormal
                                    "high"    -> VixHigh
                                    "extreme" -> VixExtreme
                                    else      -> NovaNeutralGray
                                }
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // ── Quick action buttons ──────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick  = onNavigateToRawChart,
                    modifier = Modifier.weight(1f)
                ) { Text("Raw Chart") }
                OutlinedButton(
                    onClick  = onNavigateToReliability,
                    modifier = Modifier.weight(1f)
                ) { Text("Reliability") }
            }
        }
    }
}

@Composable
private fun ConfidenceBadge(confidence: Float) {
    val color = when {
        confidence >= 80f -> NovaBuyGreen
        confidence >= 60f -> NovaWarningYellow
        else              -> NovaSellRed
    }
    Surface(
        color = color.copy(alpha = 0.2f),
        shape = MaterialTheme.shapes.small
    ) {
        Text(
            text  = "%.0f%%".format(confidence),
            style = MaterialTheme.typography.labelSmall,
            color = color,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
        )
    }
}

@Composable
private fun IndicatorChip(label: String, value: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text  = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )
        Text(
            text  = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold,
            color = color
        )
    }
}
