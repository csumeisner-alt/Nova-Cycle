package com.novacycle.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.novacycle.ui.theme.NovaSellRed
import com.novacycle.viewmodel.HealthUiState

/**
 * App-level health banners shown at the top of every data screen.
 *
 * - Red outlined card while the backend is unreachable (several consecutive
 *   failed /healthz polls).
 * - Amber card while /healthz reports status "degraded", naming the affected
 *   model(s) and alerts — mirrors the web status page banner.
 *
 * Renders nothing when the backend is healthy and reachable.
 */
@Composable
fun HealthBanners(
    state: HealthUiState,
    modifier: Modifier = Modifier
) {
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
        val health = state.health
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
    }
}
