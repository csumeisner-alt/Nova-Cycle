package com.novacycle.ui.theme

import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

/**
 * The four NovaCycle "luxe" themes. All are dark themes optimised for trading
 * dashboards; they differ in accent + background flavour. Semantic signal colors
 * (buy green / sell red / neutral) are intentionally NOT themed — gauges and
 * charts reference them directly so BUY is always green and SELL is always red.
 *
 * Unlock rules:
 *  - DARK_LUXE and MINT_LUXE are always available.
 *  - AURORA_FLUX unlocks after [unlockTaps] cumulative taps anywhere in the app.
 *  - CRIMSON_PULSE likewise, at a higher milestone.
 */
enum class NovaTheme(
    val storageKey: String,
    val displayName: String,
    val unlockTaps: Long,
    val accent: Color,
    val backgroundPreview: Color
) {
    // Storage keys are stable across releases — only the visual identity of the
    // two unlockable slots was rebranded (aurora_flux → Rose Luxe,
    // crimson_pulse → Heritage Motion).
    DARK_LUXE("dark_luxe", "Executive Gold", 0L, NovaGold, NovaBackground),
    MINT_LUXE("mint_luxe", "Mint Luxe", 0L, NovaMint, NovaBackground),
    AURORA_FLUX("aurora_flux", "Rose Luxe", 10_000L, NovaRose, NovaRoseBackground),
    CRIMSON_PULSE("crimson_pulse", "Heritage", 20_000L, NovaCopper, NovaHeritageBackground);

    val alwaysUnlocked: Boolean get() = unlockTaps == 0L

    companion object {
        val DEFAULT = DARK_LUXE
        fun fromStorageKey(key: String?): NovaTheme =
            entries.firstOrNull { it.storageKey == key } ?: DEFAULT
    }
}

private fun luxeScheme(
    primary: Color,
    tertiary: Color,
    background: Color,
    surface: Color,
    surfaceVariant: Color,
    onPrimary: Color = Color(0xFF0D0D0D)
): ColorScheme = darkColorScheme(
    primary         = primary,
    onPrimary       = onPrimary,
    secondary       = NovaSellRed,
    onSecondary     = NovaOnPrimary,
    tertiary        = tertiary,
    background      = background,
    onBackground    = NovaOnBackground,
    surface         = surface,
    onSurface       = NovaOnSurface,
    surfaceVariant  = surfaceVariant,
    error           = NovaSellRed,
    onError         = NovaOnPrimary
)

private val DarkLuxeScheme = luxeScheme(
    primary = NovaGold, tertiary = NovaGoldBright,
    background = NovaBackground, surface = NovaSurface, surfaceVariant = NovaSurfaceVariant
)

private val MintLuxeScheme = luxeScheme(
    primary = NovaMint, tertiary = NovaGold,
    background = NovaBackground, surface = NovaSurface, surfaceVariant = NovaSurfaceVariant
)

// Rose Luxe — rose-pink neon on warm near-black (occupies the aurora_flux slot)
private val RoseLuxeScheme = luxeScheme(
    primary = NovaRose, tertiary = NovaRoseGlow,
    background = NovaRoseBackground, surface = NovaRoseSurface, surfaceVariant = NovaRoseSurfaceVariant
)

// Heritage Motion — copper on warm taupe with espresso cards (crimson_pulse slot).
// The only light-background luxe theme, so it builds its scheme explicitly.
private val HeritageMotionScheme = darkColorScheme(
    primary         = NovaCopper,
    onPrimary       = Color(0xFF221709),
    secondary       = NovaSellRed,
    onSecondary     = NovaOnPrimary,
    tertiary        = NovaCopperBright,
    background      = NovaHeritageBackground,
    onBackground    = NovaHeritageOnBackground,
    surface         = NovaHeritageSurface,
    onSurface       = NovaHeritageOnSurface,
    surfaceVariant  = NovaHeritageSurfaceVariant,
    error           = NovaSellRed,
    onError         = NovaOnPrimary
)

fun NovaTheme.colorScheme(): ColorScheme = when (this) {
    NovaTheme.DARK_LUXE     -> DarkLuxeScheme
    NovaTheme.MINT_LUXE     -> MintLuxeScheme
    NovaTheme.AURORA_FLUX   -> RoseLuxeScheme
    NovaTheme.CRIMSON_PULSE -> HeritageMotionScheme
}

/** The currently applied NovaTheme, available anywhere in the Compose tree. */
val LocalNovaTheme = staticCompositionLocalOf { NovaTheme.DEFAULT }

@Composable
fun NovaCycleTheme(
    theme: NovaTheme = NovaTheme.DEFAULT,
    content: @Composable () -> Unit
) {
    CompositionLocalProvider(LocalNovaTheme provides theme) {
        MaterialTheme(
            colorScheme = theme.colorScheme(),
            typography  = NovaCycleTypography,
            content     = content
        )
    }
}
