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
import com.novacycle.ui.components.NovaLogoHeader
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
    var showConfidenceSheet by remember { mutableStateOf(false) }

    val longPred = uiState.longPrediction
    val longGaugeState = GaugeState(
        score             = longPred?.score ?: 0f,
        signal            = longPred?.signal ?: "neutral",
        confidence        = longPred?.confidence ?: 0f,
        confidencePercent = (longPred?.confidencePercent ?: 0).coerceIn(0, 100),
        trend             = longPred?.trend ?: "NEUTRAL",
        displaySignal     = longPred?.displaySignal ?: "NEUTRAL / HOLD",
        isFallback        = longPred == null || longPred.note != null,
        convictionTier    = longPred?.convictionTier,
        isCandidate       = longPred?.isCandidate ?: false,
        candidateSignal   = longPred?.candidateSignal,
        gaugePercent      = if (longPred == null || longPred.note != null) 0
                            else (((longPred.score + 100f) / 2f).toInt().coerceIn(0, 100)),
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
        convictionTier    = shortPred?.convictionTier,
        isCandidate       = shortPred?.isCandidate ?: false,
        candidateSignal   = shortPred?.candidateSignal,
        gaugePercent      = if (shortPred == null || shortPred.note != null) 0
                            else (((shortPred.score + 100f) / 2f).toInt().coerceIn(0, 100)),
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
                horizontalArrangement = Arrangement.End
            ) {
                IconButton(onClick = { viewModel.refreshAll() }) {
                    Icon(
                        imageVector      = Icons.Filled.Refresh,
                        contentDescription = "Refresh",
                        tint             = MaterialTheme.colorScheme.primary
                    )
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

            // ── Live VOO price ─────────────────────────────────────────────
            VooPriceCard(snapshot = uiState.priceSnapshot, isError = uiState.priceError)

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
                DualGaugeWidget(
                    gaugeState = longGaugeState,
                    label      = "Long-Trend",
                    modifier   = Modifier.weight(1f),
                    onClick    = { showConfidenceSheet = true }
                )
                DualGaugeWidget(
                    gaugeState = shortGaugeState,
                    label      = "Short-Trend",
                    modifier   = Modifier.weight(1f),
                    onClick    = { showConfidenceSheet = true }
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── Tier track record: real historical performance per tier ──
            TierTrackRecordCard(
                record   = uiState.tierTrackRecord,
                isError  = uiState.tierTrackRecordError,
                isLoading = uiState.isLoading && uiState.tierTrackRecord == null && !uiState.tierTrackRecordError,
                selectedWindow = uiState.tierWindow,
                onSelectWindow = { viewModel.selectTierWindow(it) }
            )

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
                                value = indicators.vixRegime?.uppercase() ?: "N/A",
                                color = when (indicators.vixRegime?.lowercase()) {
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

    if (showConfidenceSheet) {
        ConfidenceInfoSheet(onDismiss = { showConfidenceSheet = false })
    }
}

/**
 * Compact "tier track record" panel: shows realized win rate and average
 * return per conviction tier over a selectable window, with plain-language
 * copy and a sparse-data message instead of misleading tiny-sample stats.
 */
@Composable
private fun TierTrackRecordCard(
    record: com.novacycle.data.remote.models.TierTrackRecordResponse?,
    isError: Boolean,
    isLoading: Boolean,
    selectedWindow: String,
    onSelectWindow: (String) -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .luxeRim(CardDefaults.shape),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    "Tier Track Record",
                    style      = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    val windows = record?.availableWindows ?: listOf("30d", "90d", "all")
                    windows.forEach { w ->
                        FilterChip(
                            selected = w == selectedWindow,
                            onClick  = { onSelectWindow(w) },
                            label    = {
                                Text(
                                    when (w) { "all" -> "All"; else -> w },
                                    style = MaterialTheme.typography.labelSmall
                                )
                            }
                        )
                    }
                }
            }
            Text(
                "How each tier's signals actually performed, from completed buy→sell trades. Real history, not a prediction.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 2.dp, bottom = 10.dp)
            )

            when {
                isLoading -> Text(
                    "Loading track record…",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                isError || record == null -> Text(
                    "Track record unavailable right now.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                else -> {
                    TierStatsRow(
                        label     = "High-Conviction",
                        stats     = record.highConviction,
                        minSample = record.minSampleSize,
                        highlight = true
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    TierStatsRow(
                        label     = "Opportunity",
                        stats     = record.opportunity,
                        minSample = record.minSampleSize,
                        highlight = false
                    )
                    val overall = record.overall
                    Text(
                        text = if (overall.sufficientSample && overall.winRate != null)
                            "Overall: ${(overall.winRate * 100).toInt()}% win rate over ${overall.tradeCount} trades"
                        else
                            "Overall: ${overall.tradeCount} completed trade${if (overall.tradeCount == 1) "" else "s"} so far — not enough history yet",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 10.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun TierStatsRow(
    label: String,
    stats: com.novacycle.data.remote.models.TierStats,
    minSample: Int,
    highlight: Boolean,
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text       = if (highlight) "★ $label" else label,
                style      = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color      = if (highlight) MaterialTheme.colorScheme.primary
                             else MaterialTheme.colorScheme.onSurface
            )
            Text(
                "${stats.tradeCount} trade${if (stats.tradeCount == 1) "" else "s"}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        if (stats.sufficientSample && stats.winRate != null) {
            val avg = stats.avgReturnPercent
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Text(
                    "${(stats.winRate * 100).toInt()}% win rate",
                    style      = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Bold
                )
                if (avg != null) {
                    Text(
                        "${if (avg >= 0) "+" else ""}${"%.2f".format(avg)}% avg / trade",
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (avg >= 0) NovaBuyGreen else NovaSellRed,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        } else {
            Text(
                "Not enough ${label.lowercase()} signals yet — needs at least $minSample completed trades for a reliable percentage.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun VooPriceCard(
    snapshot: com.novacycle.data.remote.models.PriceSnapshotResponse?,
    isError: Boolean,
) {
    val price = snapshot?.currentPrice
    val change = snapshot?.dayChangePercent
    val direction = snapshot?.dayDirection?.lowercase()
    val directionColor = when (direction) {
        "up" -> NovaBuyGreen
        "down" -> NovaSellRed
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }
    val arrow = when (direction) {
        "up" -> "↑"
        "down" -> "↓"
        else -> "→"
    }
    val session = snapshot?.currentSession
        ?.replace('_', ' ')
        ?.replaceFirstChar { it.uppercase() }
    val sourceLabel = when {
        snapshot?.isExtendedHours == true && session != null -> "$session · extended hours"
        session != null -> session
        else -> "market feed"
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .luxeRim(CardDefaults.shape),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "VOO · LIVE PRICE",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = when {
                        isError -> "Price feed unavailable"
                        snapshot == null -> "Reading market feed…"
                        else -> sourceLabel
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    text = price?.let { "$${"%.2f".format(it)}" } ?: "—",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = arrow,
                    style = MaterialTheme.typography.headlineSmall,
                    color = directionColor,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = change?.let { "${if (it >= 0) "+" else ""}${"%.2f".format(it)}%" } ?: "—",
                    style = MaterialTheme.typography.labelMedium,
                    color = directionColor,
                )
            }
        }
    }
}

/**
 * Bottom sheet explaining the gauge's confidence zones, trend arrow, and how
 * the display signal differs from the model's raw buy/sell signal.
 * Opened by tapping either gauge — mirrors the macro-safety chip sheet.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ConfidenceInfoSheet(onDismiss: () -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
                .padding(bottom = 28.dp)
        ) {
            Text(
                "Reading the Gauges",
                style      = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(12.dp))

            // ── The big number ───────────────────────────────────────────
            Text(
                "The big percentage",
                style      = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "The large number is the gauge's directional position: 0% is the far-left (strong sell) end, 50% is neutral, and 100% is the far-right (strong buy) end. " +
                    "The BUY / HOLD / SELL word underneath is derived from that same position (65% and above reads BUY, 35% and below reads SELL, in between reads HOLD).",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
            )

            Spacer(modifier = Modifier.height(16.dp))

            // ── Confidence zones ─────────────────────────────────────────
            Text(
                "Confidence zones",
                style      = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "The smaller \"% confidence\" line is different: it is how sure the model is about its current read — not how big a move to expect. It falls into one of three zones:",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
            )
            Spacer(modifier = Modifier.height(8.dp))
            ZoneRow(
                color = NovaSellRed,
                range = "0–30%",
                name  = "Weak",
                text  = "The model has little conviction. Treat the reading as noise."
            )
            ZoneRow(
                color = NovaWarningYellow,
                range = "31–64%",
                name  = "Uncertain",
                text  = "Mixed evidence. The lean shown could easily flip."
            )
            ZoneRow(
                color = NovaBuyGreen,
                range = "65–100%",
                name  = "Strong",
                text  = "The model's indicators broadly agree on the current lean."
            )

            Spacer(modifier = Modifier.height(16.dp))

            // ── Trend arrow ──────────────────────────────────────────────
            Text(
                "Trend arrow",
                style      = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "The arrow on the bottom line shows where the model's score has been heading:\n" +
                    "▲ UP — the score has been rising recently\n" +
                    "▼ DOWN — the score has been falling recently\n" +
                    "◆ NEUTRAL — no clear recent direction",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
            )

            Spacer(modifier = Modifier.height(16.dp))

            // ── Display signal vs buy/sell signal ────────────────────────
            Text(
                "BUY BIAS vs a buy signal",
                style      = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "BUY BIAS / SELL BIAS / NEUTRAL · HOLD describe which way the model is currently leaning — they are a display summary, not a trade instruction. " +
                    "Actual buy/sell signals go through extra checks (like the macro safety override) before anything is acted on, so a \"65% BUY BIAS\" does not mean \"buy now\".",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
            )
        }
    }
}

@Composable
private fun ZoneRow(color: Color, range: String, name: String, text: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.Top
    ) {
        androidx.compose.foundation.Canvas(
            modifier = Modifier
                .padding(top = 5.dp)
                .size(8.dp)
        ) { drawCircle(color) }
        Spacer(modifier = Modifier.width(8.dp))
        Column {
            Text(
                text       = "$name · $range",
                style      = MaterialTheme.typography.bodyMedium,
                color      = color,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text  = text,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.75f)
            )
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
        // VIX data is missing or stale — the regime shown would be unreliable
        safety.vixDataMissing || safety.vixIsStale || safety.vixRegime == null -> {
            "Macro · VIX data unavailable" to NovaNeutralGray
        }
        else -> {
            val regime = safety.vixRegime.uppercase()
            val regimeColor = when (safety.vixRegime.lowercase()) {
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
                    // ── VIX data-quality warning ───────────────────────────
                    if (safety.vixDataMissing || safety.vixIsStale || safety.vixRegime == null) {
                        val warningText = when {
                            safety.vixDataMissing ->
                                "No VIX data is stored. The macro regime shown above is unavailable — VIX NORMAL is not being assumed."
                            safety.vixIsStale ->
                                "VIX data is stale (${safety.vixStalenessHours?.toInt() ?: "?"}h old). The displayed regime may not reflect current market conditions."
                            else ->
                                "VIX regime is unknown. Macro sensitivity may be degraded."
                        }
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = NovaWarningYellow.copy(alpha = 0.12f)
                            ),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text     = "⚠️ $warningText",
                                style    = MaterialTheme.typography.bodySmall,
                                color    = NovaWarningYellow,
                                modifier = Modifier.padding(10.dp)
                            )
                        }
                        Spacer(modifier = Modifier.height(12.dp))
                    }

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
                    if (safety.vixStalenessHours != null) {
                        DetailRow(
                            label = "VIX data freshness",
                            value = when {
                                safety.vixIsStale ->
                                    "${safety.vixTradingDayLag ?: "?"} trading days behind"
                                safety.vixTradingDayLag == 0 ->
                                    "Current through latest market day"
                                safety.vixTradingDayLag != null ->
                                    "${safety.vixTradingDayLag} trading days behind"
                                else ->
                                    "Current through latest market day"
                            }
                        )
                    }

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
