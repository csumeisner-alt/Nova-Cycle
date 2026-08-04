package com.novacycle.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.novacycle.data.remote.models.HealthzResponse
import com.novacycle.data.remote.models.ModelHealth
import com.novacycle.ui.theme.NovaSellRed
import com.novacycle.viewmodel.HealthUiState
import java.time.Instant
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeParseException

private val Amber = Color(0xFFFFB300)

/**
 * App-level health banners shown at the top of every data screen.
 *
 * - Red outlined card while the backend is unreachable (several consecutive
 *   failed /healthz polls).
 * - Amber card while /healthz reports status "degraded", naming the affected
 *   model(s) and alerts — mirrors the web status page banner. Tapping it opens
 *   a bottom sheet with full per-model health details (last training
 *   success/error, last trained time, neutral-fallback flag) and active alerts.
 *
 * Renders nothing when the backend is healthy and reachable.
 */
@Composable
fun HealthBanners(
    state: HealthUiState,
    modifier: Modifier = Modifier
) {
    var showDetailSheet by remember { mutableStateOf(false) }

    Column(modifier = modifier) {
        // ── Backend-unreachable notice ───────────────────────────────
        if (state.backendUnreachable) {
            OutlinedCard(
                colors = CardDefaults.outlinedCardColors(
                    containerColor = NovaSellRed.copy(alpha = 0.08f)
                ),
                border = BorderStroke(1.dp, NovaSellRed),
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
                    val now = rememberTickingNow()
                    val lastSuccess = state.lastSuccessAtMillis
                    Text(
                        text = if (lastSuccess != null) {
                            "Backend unreachable — data may be stale. " +
                                "Last updated ${formatRelativeAge(now, lastSuccess)}."
                        } else {
                            "Backend unreachable — no data received yet."
                        },
                        style = MaterialTheme.typography.bodyMedium,
                        color = NovaSellRed.copy(alpha = 0.9f)
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
        }

        // ── Degraded-predictions warning banner ──────────────────────
        val health = state.health
        if (health?.isDegraded == true) {
            val degradedModels = health.degradedModels
            OutlinedCard(
                colors = CardDefaults.cardColors(
                    containerColor = Amber.copy(alpha = 0.15f)
                ),
                border = BorderStroke(1.dp, Amber.copy(alpha = 0.45f)),
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { showDetailSheet = true }
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text       = "⚠",
                        style      = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color      = Amber,
                        modifier   = Modifier.padding(end = 8.dp)
                    )
                    Text(
                        text = if (degradedModels.isNotEmpty()) {
                            "Predictions degraded · ${degradedModels.joinToString(", ")}"
                        } else {
                            "Predictions degraded"
                        },
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.SemiBold,
                        color = Amber.copy(alpha = 0.95f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = "Details",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = Amber.copy(alpha = 0.75f),
                        modifier = Modifier.padding(start = 8.dp)
                    )
                    androidx.compose.material3.Icon(
                        imageVector = Icons.Filled.ExpandMore,
                        contentDescription = "Expand prediction health details",
                        tint = Amber.copy(alpha = 0.9f),
                        modifier = Modifier.padding(start = 2.dp)
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            // Keep the full payload out of the layout. It belongs in the
            // expandable sheet so a degraded state never consumes the screen.
            if (showDetailSheet) {
                HealthDetailSheet(health = health, onDismiss = { showDetailSheet = false })
            }
        }
    }
}

/**
 * Bottom sheet with the full /healthz payload: per-model training status
 * (last training success/error, last trained time, neutral-fallback flag)
 * and the list of active alerts.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun HealthDetailSheet(
    health: HealthzResponse,
    onDismiss: () -> Unit
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
                .padding(bottom = 24.dp)
        ) {
            Text(
                text       = "Backend health",
                style      = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text  = "Status: ${health.status}",
                style = MaterialTheme.typography.bodyMedium,
                color = if (health.isDegraded) Amber else MaterialTheme.colorScheme.onSurfaceVariant
            )

            // ── Active alerts ────────────────────────────────────────
            val alerts = health.alerts.orEmpty()
            if (alerts.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text       = "Active alerts",
                    style      = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(4.dp))
                alerts.forEach { alert ->
                    Text(
                        text  = "• $alert",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Amber.copy(alpha = 0.9f)
                    )
                }
            }

            // ── Per-model status ─────────────────────────────────────
            val models = health.models.orEmpty()
            if (models.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text       = "Models",
                    style      = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )
                models.entries.forEachIndexed { index, (name, model) ->
                    if (index > 0) {
                        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                    } else {
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                    ModelHealthRow(name = name, model = model)
                }
            } else {
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text  = "No per-model health reported.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun ModelHealthRow(name: String, model: ModelHealth) {
    val degraded = model.neutralFallback == true || model.lastTrainingSuccess == false
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text       = name,
                style      = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text       = if (degraded) "Degraded" else "OK",
                style      = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color      = if (degraded) Amber else MaterialTheme.colorScheme.primary
            )
        }
        Spacer(modifier = Modifier.height(2.dp))

        val lastTrainingSuccess = model.lastTrainingSuccess
        if (lastTrainingSuccess != null) {
            DetailLine(
                label = "Last training",
                value = if (lastTrainingSuccess) "Succeeded" else "Failed"
            )
        }
        model.lastTrainingError?.takeIf { it.isNotBlank() }?.let { error ->
            DetailLine(label = "Training error", value = error, valueColor = NovaSellRed)
        }
        model.lastTrainedAt?.takeIf { it.isNotBlank() }?.let { trainedAt ->
            val now = rememberTickingNow()
            DetailLine(label = "Last trained", value = formatTrainedAt(now, trainedAt))
        }
        if (model.neutralFallback == true) {
            DetailLine(
                label = "Neutral fallback",
                value = "Active — predictions pinned to neutral (0.5)",
                valueColor = Amber
            )
        }
    }
}

@Composable
private fun DetailLine(
    label: String,
    value: String,
    valueColor: Color = MaterialTheme.colorScheme.onSurface
) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Text(
            text  = "$label: ",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text  = value,
            style = MaterialTheme.typography.bodySmall,
            color = valueColor
        )
    }
}

/**
 * Renders the backend's ISO-8601 `last_trained_at` as a relative age
 * ("3 h 12 min ago"); falls back to the raw string when unparseable.
 *
 * Accepts instants with a Z suffix ("2026-07-25T10:00:00Z"), explicit UTC
 * offsets ("2026-07-25T12:00:00+02:00"), and naive timestamps without any
 * zone info as emitted by Python's `datetime.isoformat()`
 * ("2026-07-25T10:00:00.123456") — naive timestamps are treated as UTC.
 */
internal fun formatTrainedAt(nowMillis: Long, isoTimestamp: String): String {
    val millis = parseTrainedAtMillis(isoTimestamp) ?: return isoTimestamp
    return "${formatRelativeAge(nowMillis, millis)} ($isoTimestamp)"
}

/** Epoch millis for a backend timestamp, or null when unparseable. */
internal fun parseTrainedAtMillis(isoTimestamp: String): Long? {
    val text = isoTimestamp.trim()
    // Z-suffixed or otherwise instant-formatted.
    try {
        return Instant.parse(text).toEpochMilli()
    } catch (_: DateTimeParseException) { }
    // Explicit offset, e.g. "+02:00".
    try {
        return OffsetDateTime.parse(text).toInstant().toEpochMilli()
    } catch (_: DateTimeParseException) { }
    // Naive (no zone) — Python datetime.isoformat(); treat as UTC.
    return try {
        LocalDateTime.parse(text).toInstant(ZoneOffset.UTC).toEpochMilli()
    } catch (_: DateTimeParseException) {
        null
    }
}
