package com.novacycle.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.domain.model.*
import com.novacycle.viewmodel.ConnectionTestState
import com.novacycle.viewmodel.SettingsViewModel

/**
 * Settings screen — configures signal sensitivity, UI preferences, and backend URL.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(viewModel: SettingsViewModel = hiltViewModel()) {
    val settings         by viewModel.settings.collectAsStateWithLifecycle()
    val connTestState    by viewModel.connectionTestState.collectAsStateWithLifecycle()
    var apiUrlDraft      by remember(settings.apiBaseUrl) { mutableStateOf(settings.apiBaseUrl) }
    var apiUrlError      by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text("Settings", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(4.dp))

        SettingsSection("Signal Sensitivity") {
            Text("BUY Threshold: ${settings.buyThreshold}%", style = MaterialTheme.typography.bodyMedium)
            Slider(value = settings.buyThreshold.toFloat(), onValueChange = { viewModel.updateBuyThreshold(it.toInt()) }, valueRange = 50f..80f, steps = 29)

            Spacer(Modifier.height(4.dp))

            val sellDisplay = kotlin.math.abs(settings.sellThreshold)
            Text("SELL Threshold: ${sellDisplay}%", style = MaterialTheme.typography.bodyMedium)
            Slider(value = sellDisplay.toFloat(), onValueChange = { viewModel.updateSellThreshold(it.toInt()) }, valueRange = 50f..80f, steps = 29)

            Spacer(Modifier.height(4.dp))

            Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
                Text("Include Extended-Hours Signals", style = MaterialTheme.typography.bodyMedium)
                Switch(checked = settings.extendedHoursEnabled, onCheckedChange = { viewModel.updateExtendedHoursEnabled(it) })
            }
        }

        SettingsSection("Signal Weighting Mode") {
            SegmentedRow(
                options = WeightingMode.entries.map { when(it) { WeightingMode.BALANCED -> "Balanced"; WeightingMode.INDICATOR_HEAVY -> "Indicator"; WeightingMode.ML_HEAVY -> "ML-Heavy" } },
                selectedIndex = settings.weightingMode.ordinal,
                onSelect = { viewModel.updateWeightingMode(WeightingMode.entries[it]) }
            )
        }

        SettingsSection("Chart Smoothing") {
            SegmentedRow(
                options = SmoothingMode.entries.map { when(it) { SmoothingMode.RAW -> "Raw"; SmoothingMode.LIGHT -> "Light"; SmoothingMode.EMA -> "EMA"; SmoothingMode.HEAVY -> "Heavy" } },
                selectedIndex = settings.smoothingMode.ordinal,
                onSelect = { viewModel.updateSmoothingMode(SmoothingMode.entries[it]) }
            )
        }

        SettingsSection("Story Card Detail Level") {
            SegmentedRow(
                options = StoryLevel.entries.map { when(it) { StoryLevel.SIMPLE -> "Simple"; StoryLevel.ADVANCED -> "Advanced"; StoryLevel.EXPERT -> "Expert" } },
                selectedIndex = settings.storyCardLevel.ordinal,
                onSelect = { viewModel.updateStoryLevel(StoryLevel.entries[it]) }
            )
        }

        SettingsSection("Notification Sensitivity") {
            NotifSensitivity.entries.forEach { sensitivity ->
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(selected = settings.notificationSensitivity == sensitivity, onClick = { viewModel.updateNotifSensitivity(sensitivity) })
                    Spacer(Modifier.width(8.dp))
                    Column {
                        Text(when(sensitivity) { NotifSensitivity.STANDARD -> "Standard"; NotifSensitivity.HIGH -> "High"; NotifSensitivity.LOW -> "Low" }, style = MaterialTheme.typography.bodyMedium)
                        Text(when(sensitivity) { NotifSensitivity.STANDARD -> "Notify when crossing threshold"; NotifSensitivity.HIGH -> "Notify for signals ≥ 50%"; NotifSensitivity.LOW -> "Notify for strong signals ≥ 85%" },
                            style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                    }
                }
            }
            Spacer(Modifier.height(4.dp))
            Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
                Text("Extended-Hours Notifications", style = MaterialTheme.typography.bodyMedium)
                Switch(checked = settings.extendedHoursNotifications, onCheckedChange = { viewModel.updateExtendedHoursNotifications(it) })
            }
        }

        SettingsSection("Backend API") {
            OutlinedTextField(
                value = apiUrlDraft,
                onValueChange = {
                    apiUrlDraft = it
                    apiUrlError = null                         // clear error on edit
                    viewModel.resetConnectionTestState()
                },
                label = { Text("API Base URL") },
                placeholder = { Text("http://10.0.2.2:8080/api/") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                isError = apiUrlError != null,
                supportingText = apiUrlError?.let { err ->
                    { Text(err, color = MaterialTheme.colorScheme.error) }
                }
            )
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    onClick = { viewModel.testConnection(apiUrlDraft) },
                    modifier = Modifier.weight(1f),
                    enabled = connTestState !is ConnectionTestState.Testing
                ) {
                    Text(if (connTestState is ConnectionTestState.Testing) "Testing…" else "Test")
                }
                Button(
                    onClick = {
                        val err = viewModel.updateApiBaseUrl(apiUrlDraft)
                        apiUrlError = err
                    },
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Save")
                }
            }
            // Connection test result banner
            when (val state = connTestState) {
                is ConnectionTestState.Success -> {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        state.message,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
                is ConnectionTestState.Failure -> {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        state.message,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.error
                    )
                }
                else -> Unit
            }
        }

        Spacer(Modifier.height(8.dp))
        OutlinedButton(onClick = {
            viewModel.resetToDefaults()
            apiUrlDraft = "http://10.0.2.2:8080/api/"
            apiUrlError = null
            viewModel.resetConnectionTestState()
        }, modifier = Modifier.fillMaxWidth()) {
            Text("Reset to Defaults")
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun SettingsSection(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(10.dp))
            content()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SegmentedRow(options: List<String>, selectedIndex: Int, onSelect: (Int) -> Unit) {
    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
        options.forEachIndexed { index, label ->
            SegmentedButton(
                selected = selectedIndex == index,
                onClick  = { onSelect(index) },
                shape    = SegmentedButtonDefaults.itemShape(index = index, count = options.size),
                label    = { Text(label, style = MaterialTheme.typography.labelSmall) }
            )
        }
    }
}
