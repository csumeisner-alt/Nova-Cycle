package com.novacycle.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.ui.theme.*
import com.novacycle.viewmodel.HoldTimeViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HoldTimeScreen(
    viewModel: HoldTimeViewModel = hiltViewModel(),
    onBack: () -> Unit = {}
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Hold Time Estimate") },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
                actions = { IconButton(onClick = { viewModel.loadHoldTime() }) { Icon(Icons.Filled.Refresh, "Refresh") } }
            )
        }
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding), Alignment.Center) {
            when {
                uiState.isLoading -> CircularProgressIndicator()

                uiState.error != null -> Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("⚠️ ${uiState.error}", color = NovaSellRed)
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = { viewModel.loadHoldTime() }) { Text("Retry") }
                }

                uiState.holdTime != null -> {
                    val ht = uiState.holdTime!!
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {

                        Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                            Column(
                                modifier = Modifier.fillMaxWidth().padding(24.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                Text("Expected Hold Time", style = MaterialTheme.typography.titleMedium,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
                                Spacer(Modifier.height(12.dp))
                                Text(ht.humanReadable, fontSize = 36.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
                                Spacer(Modifier.height(4.dp))
                                Text("${ht.minutes} minutes", style = MaterialTheme.typography.bodyLarge)
                            }
                        }

                        val confColor = when { ht.confidence >= 80f -> NovaBuyGreen; ht.confidence >= 60f -> NovaWarningYellow; else -> NovaSellRed }
                        Surface(color = confColor.copy(alpha = 0.2f), shape = MaterialTheme.shapes.medium) {
                            Text("Confidence: ${"%.0f".format(ht.confidence)}%", fontWeight = FontWeight.Bold, color = confColor,
                                modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp), style = MaterialTheme.typography.titleMedium)
                        }

                        Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                            Column(Modifier.padding(16.dp)) {
                                Text("Reasoning", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                                Spacer(Modifier.height(8.dp))
                                Text(ht.reasoning, style = MaterialTheme.typography.bodyMedium)
                            }
                        }

                        Button(onClick = { viewModel.loadHoldTime() }, modifier = Modifier.fillMaxWidth()) {
                            Icon(Icons.Filled.Refresh, null)
                            Spacer(Modifier.width(8.dp))
                            Text("Recalculate")
                        }
                    }
                }

                else -> Button(onClick = { viewModel.loadHoldTime() }) { Text("Load Estimate") }
            }
        }
    }
}
