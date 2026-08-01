package com.novacycle.ui.screens

import android.content.ContextWrapper
import androidx.activity.ComponentActivity
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.domain.model.*
import com.novacycle.ui.theme.NovaTheme
import com.novacycle.viewmodel.ConnectionTestState
import com.novacycle.viewmodel.SettingsViewModel
import com.novacycle.viewmodel.ThemeViewModel
import kotlinx.coroutines.delay

/**
 * Settings screen — configures signal sensitivity, UI preferences, and backend URL.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel = hiltViewModel(),
    // The theme VM MUST be the activity-scoped instance: MainActivity registers
    // global taps (and emits unlock events) on that instance, so scoping this to
    // the nav destination would silently split the event stream in two.
    themeViewModel: ThemeViewModel = activityScopedThemeViewModel()
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

        SettingsSection("Appearance") {
            ThemePicker(themeViewModel)
        }

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
            Spacer(Modifier.height(4.dp))
            Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("High-Conviction Signals Only", style = MaterialTheme.typography.bodyMedium)
                    Text("Only notify for signals that pass all confirmation checks",
                        style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
                }
                Switch(checked = settings.highConvictionOnly, onCheckedChange = { viewModel.updateHighConvictionOnly(it) })
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
 * Resolves the activity-scoped [ThemeViewModel] — the same instance MainActivity
 * uses for tap counting — regardless of the nav destination we are composed in.
 */
@Composable
private fun activityScopedThemeViewModel(): ThemeViewModel {
    val context = LocalContext.current
    val activity = remember(context) {
        generateSequence(context) { (it as? ContextWrapper)?.baseContext }
            .filterIsInstance<ComponentActivity>()
            .first()
    }
    return hiltViewModel(activity)
}

/**
 * Theme picker: one swatch per luxe theme. Locked themes are greyed out and
 * non-selectable (deliberately without a lock icon). Newly unlocked themes get
 * a brief gold shimmer sweep — the sound + haptic play globally.
 */
@Composable
private fun ThemePicker(themeViewModel: ThemeViewModel) {
    val themeState by themeViewModel.themeState.collectAsStateWithLifecycle()

    // Themes unlocked within the last few seconds → shimmer
    var shimmering by remember { mutableStateOf(setOf<NovaTheme>()) }
    LaunchedEffect(themeViewModel) {
        themeViewModel.unlockEvents.collect { theme ->
            shimmering = shimmering + theme
            delay(4000)
            shimmering = shimmering - theme
        }
    }

    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        NovaTheme.entries.forEach { theme ->
            val unlocked = themeState.isUnlocked(theme)
            val selected = themeState.selectedTheme == theme
            ThemeSwatch(
                theme = theme,
                unlocked = unlocked,
                selected = selected,
                shimmer = theme in shimmering,
                tapCount = themeState.tapCount,
                modifier = Modifier.weight(1f),
                onClick = { if (unlocked) themeViewModel.selectTheme(theme) }
            )
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ThemeSwatch(
    theme: NovaTheme,
    unlocked: Boolean,
    selected: Boolean,
    shimmer: Boolean,
    tapCount: Long,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val shape = RoundedCornerShape(12.dp)
    val accent = if (unlocked) theme.accent else Color(0xFF3A3A3A)
    val bg = if (unlocked) theme.backgroundPreview else Color(0xFF1C1C1C)

    // Long-pressing a locked swatch briefly reveals the exact tap progress.
    var showProgressText by remember { mutableStateOf(false) }
    LaunchedEffect(showProgressText) {
        if (showProgressText) {
            delay(3000)
            showProgressText = false
        }
    }

    val progress = if (unlocked || theme.unlockTaps <= 0L) 1f
                   else (tapCount.toFloat() / theme.unlockTaps).coerceIn(0f, 1f)

    Column(
        modifier = modifier
            .clip(shape)
            .background(bg)
            .border(
                width = if (selected) 2.dp else 1.dp,
                color = when {
                    selected -> theme.accent
                    unlocked -> Color(0xFF2E2E2E)
                    else     -> Color(0xFF232323)
                },
                shape = shape
            )
            .then(
                if (unlocked) Modifier.clickable(onClick = onClick)
                else Modifier.combinedClickable(
                    onClick = { /* locked — non-selectable */ },
                    onLongClick = { showProgressText = true }
                )
            )
            .then(if (shimmer) Modifier.shimmerOverlay() else Modifier)
            .padding(vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Box(contentAlignment = Alignment.Center) {
            // Thin progress ring around the accent dot for locked themes —
            // a subtle hint of how close the unlock milestone is (no lock icon).
            if (!unlocked) {
                Canvas(Modifier.size(34.dp)) {
                    val stroke = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round)
                    drawArc(
                        color = Color(0xFF2E2E2E),
                        startAngle = -90f, sweepAngle = 360f,
                        useCenter = false, style = stroke
                    )
                    if (progress > 0f) {
                        drawArc(
                            color = theme.accent.copy(alpha = 0.85f),
                            startAngle = -90f, sweepAngle = 360f * progress,
                            useCenter = false, style = stroke
                        )
                    }
                }
            }
            Box(
                Modifier
                    .size(26.dp)
                    .clip(CircleShape)
                    .background(accent)
            )
        }
        Text(
            text = if (!unlocked && showProgressText)
                       "%,d / %,d".format(tapCount, theme.unlockTaps)
                   else theme.displayName,
            style = MaterialTheme.typography.labelSmall,
            color = if (unlocked) MaterialTheme.colorScheme.onSurface
                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f),
            maxLines = 1
        )
    }
}

/** Repeating diagonal gold shimmer sweep drawn over the swatch content. */
@Composable
private fun Modifier.shimmerOverlay(): Modifier {
    val transition = rememberInfiniteTransition(label = "shimmer")
    val x by transition.animateFloat(
        initialValue = -1f,
        targetValue = 2f,
        animationSpec = infiniteRepeatable(tween(900, easing = LinearEasing), RepeatMode.Restart),
        label = "shimmerX"
    )
    return this.background(
        Brush.linearGradient(
            colors = listOf(
                Color.Transparent,
                Color(0xFFFFD700).copy(alpha = 0.45f),
                Color.Transparent
            ),
            start = Offset(x * 300f, x * 120f),
            end = Offset(x * 300f + 160f, x * 120f + 80f)
        )
    )
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
