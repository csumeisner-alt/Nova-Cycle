package com.novacycle.ui.theme

import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

/**
 * NovaCycle is dark-only, optimised for trading dashboards. Four accent
 * palettes are available (see [AppTheme]); the selected theme swaps the
 * accent (primary/tertiary) colors while backgrounds stay near-black and
 * trading semantics (BUY green / SELL red via [NovaBuyGreen]/[NovaSellRed])
 * remain identical across all themes.
 */
private fun novaScheme(primary: androidx.compose.ui.graphics.Color, tertiary: androidx.compose.ui.graphics.Color): ColorScheme =
    darkColorScheme(
        primary         = primary,
        onPrimary       = NovaBackground,
        secondary       = NovaSellRed,
        onSecondary     = NovaOnPrimary,
        tertiary        = tertiary,
        background      = NovaBackground,
        onBackground    = NovaOnBackground,
        surface         = NovaSurface,
        onSurface       = NovaOnSurface,
        surfaceVariant  = NovaSurfaceVariant,
        error           = NovaSellRed,
        onError         = NovaOnPrimary
    )

private val DarkLuxeScheme     = novaScheme(LuxeGold, LuxeGoldDeep)
private val AuroraFluxScheme   = novaScheme(AuroraTeal, AuroraViolet)
private val CrimsonPulseScheme = novaScheme(CrimsonAccent, CrimsonEmber)
private val MintLuxeScheme     = novaScheme(MintAccent, MintEmerald)

fun colorSchemeFor(theme: AppTheme): ColorScheme = when (theme) {
    AppTheme.DARK_LUXE     -> DarkLuxeScheme
    AppTheme.AURORA_FLUX   -> AuroraFluxScheme
    AppTheme.CRIMSON_PULSE -> CrimsonPulseScheme
    AppTheme.MINT_LUXE     -> MintLuxeScheme
}

@Composable
fun NovaCycleTheme(
    appTheme: AppTheme = AppTheme.DARK_LUXE,
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = colorSchemeFor(appTheme),
        typography  = NovaCycleTypography,
        content     = content
    )
}
