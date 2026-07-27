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
import com.novacycle.ui.components.ThemeAwareGauge
import com.novacycle.ui.components.NovaLogoHeader
import com.novacycle.ui.components.TickerSelector
import com.novacycle.ui.components.luxeRim
import com.novacycle.ui.components.formatRelativeAge
import com.novacycle.ui.components.rememberTickingNow
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

    val longPred = uiState.longPrediction
    val longGaugeState = GaugeState(
        score             = longPred?.score ?: 0f,
        signal            = longPred?.signal ?: "neutral",
        confidence        = longPred?.confidence ?: 0f,
        confidencePercent = (longPred?.confidencePercent ?: 0).coerceIn(0, 100),
        trend             = longPred?.trend ?: "NEUTRAL",
        displaySignal     = longPred?.displaySignal ?: "NEUTRAL / HOLD",
        isFallback        = longPred == null || longPred.note != null,
        gaugeType         = "long",
        ticker            = uiState.selectedTicker,
        isLoading         = uiState.isLoading && uiState.longPrediction == null
    )
    val shortPred = uiState.shortPrediction
    val shortGaugeState = GaugeState(
        score             = shortPred?.score ?: 0f,
        signal            = shortPred?.signal ?: "neutral",
        confidence        = shortPred?.confidence ?: 0f,
        confidencePercent = (shortPred?.confidencePercent ?: 0).coerceIn(0, 100),
        trend             = shortPred?.trend ?: "NEUTRAL",
        displaySignal     = shortPred?.displaySignal ?: "NEUTRAL / HOLD",
        isFallback        = shortPred == null || shortPred.note != null,
        gaugeType         = "short",
        ticker            = uiState.selectedTicker,
        isLoading         = uiState.isLoading && uiState.shortPrediction == null
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
            // ── Brand header: tappable breathing logo, top-center ─────────
            NovaLogoHeader()

            Spacer(modifier = Modifier.height(4.dp))

            // ── Controls row ──────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Spacer(modifier = Modifier.width(4.dp))
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

            // "Updated X ago" label under the header, ticking as time passes
            val lastUpdated = uiState.lastUpdatedAtMillis
            if (lastUpdated != null) {
                val now = rememberTickingNow()
                Text(
                    text     = "Updated ${formatRelativeAge(now, lastUpdated)}",
                    style    = MaterialTheme.typography.labelSmall,
                    // On the app background (not a card) — onBackground keeps it
                    // legible on Heritage's light taupe as well as dark themes.
                    color    = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 2.dp)
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // ── Macro safety chip ─────────────────────────────────────────
            MacroSafetyChip(
                safety    = uiState.macroSafety,
                isError   = uiState.macroSafetyError,
                isLoading = uiState.isLoading && uiState.macroSafety == null && !uiState.macroSafetyError
            )

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

            // NOTE: the backend-unreachable and degraded-predictions banners
            // are now rendered app-wide by HealthBanners in NavGraph, driven
            // by the shared HealthViewModel — not per-screen.

            // ── Dual Gauge widgets ────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                ThemeAwareGauge(
                    gaugeState = longGaugeState,
                    label      = "Long-Trend",
                    modifier   = Modifier.weight(1f)
                )
                ThemeAwareGauge(
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
                modifier = Modifier.fillMaxWidth().luxeRim(CardDefaults.shape),
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
                    modifier = Modifier.fillMaxWidth().luxeRim(CardDefaults.shape),
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

/**
 * Small status chip summarizing the backend's macro safety state:
 *  - override active → warning yellow with the suppression direction
 *  - safe → VIX-regime-colored "Macro OK · VIX <regime>"
 *  - loading → muted "Macro …"
 *  - error → muted "Macro unavailable" (never blocks the dashboard)
 *
 * Tapping the chip opens a bottom sheet with the full explanation
 * (reason, VIX close + regime, long score vs thresholds, last override time).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MacroSafetyChip(
    safety: com.novacycle.data.remote.models.MacroSafetyResponse?,
    isError: Boolean,
    isLoading: Boolean
) {
    var showSheet by remember { mutableStateOf(false) }
    val (label, color) = when {
        isLoading      -> "Macro …" to NovaNeutralGray
        safety == null -> "Macro unavailable" to NovaNeutralGray
        safety.overrideActive -> {
            val dir = if (safety.suppressesShortBuy) "BUYs suppressed" else "SELLs suppressed"
            "Macro override · $dir" to NovaWarningYellow
        }
        else -> {
            val regime = safety.vixRegime?.uppercase() ?: "?"
            val regimeColor = when (safety.vixRegime?.lowercase()) {
                "low"     -> VixLow
                "normal"  -> VixNormal
                "high"    -> VixHigh
                "extreme" -> VixExtreme
                else      -> NovaNeutralGray
            }
            "Macro OK · VIX $regime" to regimeColor
        }
    }
    // isError falls into the safety == null branch above; kept explicit for clarity
    if (isError && safety == null && !isLoading) { /* muted fallback already chosen */ }

    Surface(
        onClick = { showSheet = true },
        color = color.copy(alpha = 0.15f),
        shape = MaterialTheme.shapes.small,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
        ) {
            androidx.compose.foundation.Canvas(modifier = Modifier.size(8.dp)) { drawCircle(color) }
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text  = label,
                style = MaterialTheme.typography.labelMedium,
                color = color,
                fontWeight = FontWeight.SemiBold
            )
        }
    }

    if (showSheet) {
        MacroSafetyDetailSheet(
            safety      = safety,
            isLoading   = isLoading,
            statusColor = color,
            statusLabel = label,
            onDismiss   = { showSheet = false }
        )
    }
}

/**
 * Bottom sheet with the full macro safety explanation. Renders sensible
 * placeholders when data is loading or unavailable — never crashes on nulls.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MacroSafetyDetailSheet(
    safety: com.novacycle.data.remote.models.MacroSafetyResponse?,
    isLoading: Boolean,
    statusColor: Color,
    statusLabel: String,
    onDismiss: () -> Unit
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp)
                .padding(bottom = 28.dp)
        ) {
            Text(
                "Macro Safety",
                style      = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))

            // Status line — reuse the chip's color + label so both agree
            Row(verticalAlignment = Alignment.CenterVertically) {
                androidx.compose.foundation.Canvas(modifier = Modifier.size(8.dp)) { drawCircle(statusColor) }
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text       = statusLabel,
                    style      = MaterialTheme.typography.labelLarge,
                    color      = statusColor,
                    fontWeight = FontWeight.SemiBold
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            when {
                isLoading -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            strokeWidth = 2.dp
                        )
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            "Loading macro safety details…",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                }
                safety == null -> {
                    Text(
                        "Macro safety data is currently unavailable. " +
                            "The dashboard keeps working — pull to refresh to retry.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                    )
                }
                else -> {
                    // ── Reason ────────────────────────────────────────────
                    if (safety.reason.isNotBlank()) {
                        Text(
                            text  = safety.reason,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                    }

                    // ── VIX ───────────────────────────────────────────────
                    DetailRow(
                        label = "VIX close",
                        value = safety.vixClose?.let { "%.2f".format(it) } ?: "—"
                    )
                    DetailRow(
                        label = "VIX regime",
                        value = safety.vixRegime?.uppercase() ?: "—"
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    // ── Long score vs thresholds ──────────────────────────
                    DetailRow(
                        label = "Long score",
                        value = "%.1f".format(safety.longScore)
                    )
                    val t = safety.thresholds
                    if (t != null) {
                        DetailRow(
                            label = "Strong-bear threshold",
                            value = "%.1f".format(t.longStrongBear)
                        )
                        DetailRow(
                            label = "Strong-bull threshold",
                            value = "%.1f".format(t.longStrongBull)
                        )
                        DetailRow(
                            label = "ML override threshold",
                            value = "%.2f".format(t.mlOverrideThreshold)
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    // ── Override history ──────────────────────────────────
                    DetailRow(
                        label = "Last override",
                        value = safety.lastOverrideAppliedAt ?: "Never"
                    )
                    safety.computedAt?.let {
                        DetailRow(label = "Computed at", value = it)
                    }
                }
            }
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text  = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f)
        )
        Text(
            text       = value,
            style      = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold
        )
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
