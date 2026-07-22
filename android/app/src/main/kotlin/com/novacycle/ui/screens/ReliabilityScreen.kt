package com.novacycle.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.novacycle.ui.theme.NovaBuyGreen
import com.novacycle.ui.theme.NovaNeutralGray

/**
 * Reliability screen — placeholder for future trade performance metrics.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReliabilityScreen(onBack: () -> Unit = {}) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Trade Reliability") },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } }
            )
        }
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(24.dp), Alignment.CenterHorizontally) {
            Text("Trade Reliability Metrics", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
            Spacer(Modifier.height(8.dp))
            Text("Coming soon — track your BUY→SELL cycle performance", style = MaterialTheme.typography.bodyMedium, color = NovaNeutralGray, textAlign = TextAlign.Center)

            Spacer(Modifier.height(32.dp))

            val placeholders = listOf("Win Rate" to "--", "Avg Return" to "--", "Avg Hold Time" to "--", "Total Cycles" to "--", "Best Return" to "--", "Worst Return" to "--")
            placeholders.chunked(2).forEach { row ->
                Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), Arrangement.spacedBy(12.dp)) {
                    row.forEach { (label, value) -> MetricCard(label, value, Modifier.weight(1f)) }
                    if (row.size == 1) Spacer(Modifier.weight(1f))
                }
            }

            Spacer(Modifier.height(24.dp))
            Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = NovaBuyGreen.copy(alpha = 0.1f))) {
                Text(
                    "📊 Once you accumulate BUY→SELL cycles, this screen will show:\n• Win rate\n• Average return per cycle\n• Hold duration accuracy\n• Long vs Short gauge accuracy\n• Extended-hours performance",
                    style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(16.dp)
                )
            }
        }
    }
}

@Composable
private fun MetricCard(label: String, value: String, modifier: Modifier = Modifier) {
    Card(modifier = modifier, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.fillMaxWidth().padding(16.dp), Alignment.CenterHorizontally) {
            Text(value, style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, color = NovaNeutralGray)
            Spacer(Modifier.height(4.dp))
            Text(label, style = MaterialTheme.typography.labelSmall, textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
        }
    }
}
