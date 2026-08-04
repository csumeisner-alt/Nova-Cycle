package com.novacycle.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.model.StoryLevel
import com.novacycle.ui.theme.NovaBuyGreen
import com.novacycle.ui.theme.NovaSellRed
import com.novacycle.ui.theme.NovaWarningYellow

/**
 * Modal bottom sheet showing signal context in three detail levels:
 *  SIMPLE   — headline + 3 key bullets + hold time
 *  ADVANCED — adds per-indicator breakdown, ML confidence bar, gap/liquidity info
 *  EXPERT   — adds session details, macro override explanation, raw IDs
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SignalStoryCard(
    signal: SignalData,
    storyLevel: StoryLevel,
    holdTimeText: String = "",
    indicatorBreakdown: Map<String, Any> = emptyMap(),
    mlConfidence: Float = 0f,
    onDismiss: () -> Unit
) {
    val signalColor = if (signal.isBuy) NovaBuyGreen else NovaSellRed
    val signalLabel = "${signal.signalType.uppercase()} ${"%.0f".format(signal.confidence * 100f)}%"
    val gaugeLabel  = if (signal.isLongGauge) "Long-Trend" else "Short-Trend"

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor   = MaterialTheme.colorScheme.surface
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 8.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(signalLabel, style = MaterialTheme.typography.headlineMedium,
                        color = signalColor, fontWeight = FontWeight.Bold)
                    Text(gaugeLabel, style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface)
                    if (signal.isCandidate) {
                        CandidateBadge(direction = signal.signalType)
                    } else {
                        signal.convictionTier?.let { ConvictionTierBadge(it) }
                    }
                    if (signal.isDegradedModel) {
                        DegradedModelBadge(modelState = signal.modelState)
                    }
                }
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Filled.Close, contentDescription = "Close")
                }
            }

            Spacer(modifier = Modifier.height(12.dp))
            HorizontalDivider()
            Spacer(modifier = Modifier.height(12.dp))

            SimpleSummary(signal, holdTimeText)

            if (storyLevel >= StoryLevel.ADVANCED) {
                Spacer(modifier = Modifier.height(16.dp))
                AdvancedBreakdown(signal, indicatorBreakdown, mlConfidence)
            }

            if (storyLevel >= StoryLevel.EXPERT) {
                Spacer(modifier = Modifier.height(16.dp))
                ExpertDetails(signal)
            }

            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

@Composable
private fun SimpleSummary(signal: SignalData, holdTimeText: String) {
    SectionTitle("Summary")
    val bullets = buildList {
        if (signal.isCandidate) add("⚡ Candidate signal — directional hint only, not executable")
        if (signal.isExtendedHours) add("⚡ Extended-hours session signal")
        if (signal.gapType != null) add("📊 Gap detected: ${signal.gapType}")
        if (signal.liquidityScore < 0.5f) add("💧 Low liquidity environment")
        if (signal.macroOverrideApplied) add("⚠️ Macro override was applied")
        if (signal.isDegradedModel) add("🛑 Stored while the model was degraded — treat with caution")
        if (signal.isHighConviction) add("⭐ High-conviction signal — all confirmation checks passed")
        add("📈 Confidence: ${"%.1f".format(signal.confidence * 100f)}%")
    }
    bullets.forEach { bullet ->
        Text(bullet, style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.padding(vertical = 3.dp))
    }
    if (holdTimeText.isNotBlank()) {
        Spacer(modifier = Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("⏱ Hold time estimate: ", style = MaterialTheme.typography.bodyMedium)
            Text(holdTimeText, style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.primary)
        }
    }
}

@Composable
private fun AdvancedBreakdown(signal: SignalData, indicatorBreakdown: Map<String, Any>, mlConfidence: Float) {
    SectionTitle("Indicator Breakdown")
    // The backend may include textual annotations (e.g. "liquidity_adjustment") alongside
    // numeric scores. Only numeric entries are displayed in the sorted bar list.
    val numericBreakdown = indicatorBreakdown.mapNotNull { (name, value) ->
        val score = (value as? Number)?.toFloat() ?: return@mapNotNull null
        name to score
    }.sortedByDescending { it.second }

    if (numericBreakdown.isEmpty()) {
        Text("No indicator breakdown available", style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
    } else {
        numericBreakdown.forEach { (name, score) ->
            val scoreColor = if (score > 0) NovaBuyGreen else NovaSellRed
            Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), Arrangement.SpaceBetween) {
                Text(name, style = MaterialTheme.typography.bodyMedium)
                Text("%+.1f".format(score), style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold, color = scoreColor)
            }
        }
    }
    Spacer(modifier = Modifier.height(12.dp))
    SectionTitle("ML Confidence")
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        LinearProgressIndicator(
            progress = { (mlConfidence / 100f).coerceIn(0f, 1f) },
            modifier = Modifier.weight(1f).height(8.dp),
            color = NovaBuyGreen, trackColor = MaterialTheme.colorScheme.surfaceVariant
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text("%.0f%%".format(mlConfidence), style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
    }
    if (signal.gapType != null) {
        Spacer(modifier = Modifier.height(8.dp))
        Text("Gap ${signal.gapType} detected — affects BUY score weighting",
            style = MaterialTheme.typography.bodyMedium, color = NovaWarningYellow)
    }
    if (signal.liquidityScore < 0.5f) {
        Spacer(modifier = Modifier.height(4.dp))
        Text("Low liquidity (score: ${"%.2f".format(signal.liquidityScore)}) — filtered accordingly",
            style = MaterialTheme.typography.bodyMedium, color = Color.Gray)
    }
}

@Composable
private fun ExpertDetails(signal: SignalData) {
    SectionTitle("Expert Details")
    listOf(
        "Session Type"   to signal.sessionType.replaceFirstChar { it.uppercase() },
        "Extended Hours" to if (signal.isExtendedHours) "Yes" else "No",
        "Gap Type"       to (signal.gapType ?: "None"),
        "Liquidity"      to "%.3f".format(signal.liquidityScore),
        "Macro Override" to if (signal.macroOverrideApplied) "Applied" else "Not applied",
        "Model State"    to (signal.modelState ?: "Unknown (pre-tracking)"),
        "Conviction"     to when {
            signal.isCandidate                    -> "Candidate (not executable)"
            signal.convictionTier == "high_conviction" -> "High-Conviction"
            signal.convictionTier == "opportunity"     -> "Opportunity"
            else                                  -> "—"
        },
        "Cycle ID"       to (signal.cycleId?.take(8) ?: "—"),
        "Signal ID"      to signal.id.take(8),
        "Timestamp"      to signal.timestamp
    ).forEach { (key, value) ->
        Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), Arrangement.SpaceBetween) {
            Text(key, style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
            Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
        }
    }
    if (signal.macroOverrideApplied) {
        Spacer(modifier = Modifier.height(8.dp))
        Card(colors = CardDefaults.cardColors(containerColor = NovaWarningYellow.copy(alpha = 0.15f))) {
            Text(
                "⚠️ Macro override active: Long-trend score exceeded threshold. Short-term BUY suppressed unless ML confidence > 80%.",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(10.dp),
                color = NovaWarningYellow
            )
        }
    }
    if (signal.convictionReasons.isNotEmpty()) {
        Spacer(modifier = Modifier.height(8.dp))
        SectionTitle("Conviction Analysis")
        signal.convictionReasons.forEach { reason ->
            Text(reason, style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f),
                modifier = Modifier.padding(vertical = 2.dp))
        }
    }
}

/** Small pill badge indicating the signal's conviction tier. */
@Composable
fun ConvictionTierBadge(tier: String, modifier: Modifier = Modifier) {
    val isHigh = tier == "high_conviction"
    val label = if (isHigh) "⭐ HIGH-CONVICTION" else "OPPORTUNITY"
    val bg = if (isHigh) NovaWarningYellow.copy(alpha = 0.20f)
             else MaterialTheme.colorScheme.surfaceVariant
    val fg = if (isHigh) NovaWarningYellow
             else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
    Surface(
        modifier = modifier.padding(top = 4.dp),
        shape = MaterialTheme.shapes.small,
        color = bg
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            color = fg,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
        )
    }
}

/**
 * Badge shown when a stored signal was generated under a non-healthy model
 * state (training-stuck, stale rollback, model unavailable, or baseline mode).
 *
 * Baseline mode uses an amber tint with distinct copy matching the web
 * dashboard ("BASELINE MODE · NO TRAINED EDGE") — it is a known-good
 * calibrated fallback, not an error, so it must not look like a red alert.
 */
@Composable
fun DegradedModelBadge(modelState: String? = null, modifier: Modifier = Modifier) {
    val isBaseline = modelState == "baseline_mode"
    val badgeColor = if (isBaseline) Color(0xFFE65100) else NovaSellRed
    val label = if (isBaseline) "BASELINE MODE · NO TRAINED EDGE" else "⚠ DEGRADED MODEL"
    Surface(
        modifier = modifier.padding(top = 4.dp),
        shape = MaterialTheme.shapes.small,
        color = badgeColor.copy(alpha = 0.15f)
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            color = badgeColor,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
        )
    }
}

/**
 * Badge shown when a signal is a candidate — the raw gauge crossed its
 * threshold but current conditions make it non-executable.
 */
@Composable
fun CandidateBadge(direction: String, modifier: Modifier = Modifier) {
    val dirLabel = direction.uppercase()
    val bg = NovaWarningYellow.copy(alpha = 0.12f)
    val fg = NovaWarningYellow.copy(alpha = 0.85f)
    Surface(
        modifier = modifier.padding(top = 4.dp),
        shape = MaterialTheme.shapes.small,
        color = bg
    ) {
        Text(
            "⚡ $dirLabel CANDIDATE",
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            color = fg,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
        )
    }
}

@Composable
private fun SectionTitle(title: String) {
    Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold,
        color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(bottom = 6.dp))
}

private operator fun StoryLevel.compareTo(other: StoryLevel): Int = ordinal.compareTo(other.ordinal)
