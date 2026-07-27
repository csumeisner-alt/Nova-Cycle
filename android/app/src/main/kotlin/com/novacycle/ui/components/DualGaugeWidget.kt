package com.novacycle.ui.components

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.novacycle.domain.model.GaugeState

/**
 * Compatibility entry point retained for callers outside the dashboard.
 * The implementation is now fully theme-aware.
 */
@Composable
fun DualGaugeWidget(
    gaugeState: GaugeState,
    label: String,
    modifier: Modifier = Modifier
) = ThemeAwareGauge(gaugeState = gaugeState, label = label, modifier = modifier)