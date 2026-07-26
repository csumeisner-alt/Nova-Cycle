package com.novacycle.ui.screens

import android.app.Activity
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.billing.MintBillingState
import com.novacycle.domain.model.*
import com.novacycle.domain.theme.ThemeUnlockLogic
import com.novacycle.ui.theme.AppTheme
import com.novacycle.ui.theme.colorSchemeFor
import com.novacycle.viewmodel.ConnectionTestState
import com.novacycle.viewmodel.SettingsViewModel
import com.novacycle.viewmodel.ThemeViewModel

/**
 * Settings screen — configures signal sensitivity, UI preferences, and backend URL.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel = hiltViewModel(),
    themeViewModel: ThemeViewModel = hiltViewModel()
) {
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

        AppearanceSection(themeViewModel)

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
                placeholder = { Text(com.novacycle.BuildConfig.API_BASE_URL) },
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
            apiUrlDraft = com.novacycle.BuildConfig.API_BASE_URL
            apiUrlError = null
            viewModel.resetConnectionTestState()
        }, modifier = Modifier.fillMaxWidth()) {
            Text("Reset to Defaults")
        }
        Spacer(Modifier.height(24.dp))
    }
}

/**
 * Theme picker: Dark Luxe (always available), Aurora Flux & Crimson Pulse
 * (20,000-tap achievement), and Mint Luxe (Play Billing purchase).
 */
@Composable
private fun AppearanceSection(themeViewModel: ThemeViewModel) {
    val themeState   by themeViewModel.themeState.collectAsStateWithLifecycle()
    val billingState by themeViewModel.billingState.collectAsStateWithLifecycle()
    val activity = LocalContext.current as? Activity
    var billingMessage by remember { mutableStateOf<String?>(null) }

    SettingsSection("Appearance") {
        AppTheme.entries.forEach { theme ->
            val available = ThemeUnlockLogic.isThemeAvailable(
                theme, themeState.auroraUnlocked, themeState.crimsonUnlocked, themeState.mintUnlocked
            )
            val selected = themeState.selectedTheme == theme
            Row(
                Modifier.fillMaxWidth().padding(vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                RadioButton(
                    selected = selected,
                    onClick = { themeViewModel.selectTheme(theme) },
                    enabled = available
                )
                // Accent swatch preview
                Box(
                    Modifier
                        .size(18.dp)
                        .clip(CircleShape)
                        .background(colorSchemeFor(theme).primary)
                )
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        theme.displayName,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                        color = if (available) MaterialTheme.colorScheme.onSurface
                                else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                    )
                    Text(
                        theme.tagline,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                }
                if (!available) {
                    when (theme) {
                        AppTheme.MINT_LUXE -> MintBuyButton(
                            billingState = billingState,
                            onBuy = {
                                val act = activity
                                billingMessage = when {
                                    act == null -> "Can't start purchase outside an activity"
                                    themeViewModel.purchaseMintLuxe(act) -> null
                                    else -> "Google Play Billing isn't available right now"
                                }
                            }
                        )
                        else -> Icon(
                            Icons.Filled.Lock,
                            contentDescription = "Locked",
                            tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
                        )
                    }
                }
            }
        }

        // Tap-achievement progress toward Aurora Flux + Crimson Pulse
        if (!themeState.auroraUnlocked || !themeState.crimsonUnlocked) {
            Spacer(Modifier.height(8.dp))
            Text(
                "Logo tap achievement: ${ThemeUnlockLogic.progressLabel(themeState.tapCount)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
            Spacer(Modifier.height(4.dp))
            LinearProgressIndicator(
                progress = { ThemeUnlockLogic.progressFraction(themeState.tapCount) },
                modifier = Modifier.fillMaxWidth()
            )
            Text(
                "Tap the gold logo on the dashboard 20,000 times to unlock Aurora Flux and Crimson Pulse.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                modifier = Modifier.padding(top = 4.dp)
            )
        }

        // Billing status / error feedback
        val statusText = billingMessage ?: when (val s = billingState) {
            is MintBillingState.Unavailable ->
                if (!themeState.mintUnlocked) s.reason else null
            MintBillingState.Pending -> "Purchase in progress…"
            else -> null
        }
        if (statusText != null) {
            Spacer(Modifier.height(6.dp))
            Text(
                statusText,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
        }
    }
}

@Composable
private fun MintBuyButton(billingState: MintBillingState, onBuy: () -> Unit) {
    val (label, enabled) = when (billingState) {
        is MintBillingState.Available -> "Buy ${billingState.formattedPrice}" to true
        MintBillingState.Pending      -> "Pending…" to false
        else                          -> "Buy $1.49" to false
    }
    Button(onClick = onBuy, enabled = enabled, contentPadding = PaddingValues(horizontal = 12.dp)) {
        Text(label, style = MaterialTheme.typography.labelSmall)
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
