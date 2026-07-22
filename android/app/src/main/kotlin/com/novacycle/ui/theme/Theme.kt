package com.novacycle.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

/**
 * NovaCycle uses a dark-only theme optimised for trading dashboards.
 * High-contrast green/red against near-black background reduces eye strain
 * when monitoring signals during extended sessions.
 */
private val NovaDarkColorScheme = darkColorScheme(
    primary         = NovaBuyGreen,
    onPrimary       = NovaOnPrimary,
    secondary       = NovaSellRed,
    onSecondary     = NovaOnPrimary,
    tertiary        = NovaExtendedBlue,
    background      = NovaBackground,
    onBackground    = NovaOnBackground,
    surface         = NovaSurface,
    onSurface       = NovaOnSurface,
    surfaceVariant  = NovaSurfaceVariant,
    error           = NovaSellRed,
    onError         = NovaOnPrimary
)

@Composable
fun NovaCycleTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = NovaDarkColorScheme,
        typography  = NovaCycleTypography,
        content     = content
    )
}
