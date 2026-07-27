package com.novacycle.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * Ambient background treatment rendered behind every screen.
 * Each luxe theme picks one so pages feel alive without changing screen code.
 */
enum class AmbientStyle {
    /** Executive Gold — drifting warm ember glow + faint monogram lattice. */
    EMBER_GLOW,
    /** Rose Luxe — flowing rose-pink neon light ribbons. */
    LIGHT_RIBBONS,
    /** Mint Luxe — soft teal aurora wisps sweeping the edges. */
    AURORA_WISPS,
    /** Heritage Motion — warm monogram pattern + central green/red racing stripe. */
    HERITAGE_PATTERN
}

enum class GaugeMotionStyle {
    CONFIDENT_SWEEP,
    FLUID_WAVE,
    ELEGANT_SWIRL,
    REFINED_GLIDE
}

data class GaugePalette(
    val arcStart: Color,
    val arcEnd: Color,
    val buy: Color,
    val sell: Color,
    val hold: Color,
    val needle: Color,
    val label: Color,
    val glow: Color,
    val background: Color,
    val motion: GaugeMotionStyle
)

/**
 * Per-theme visual parameters beyond the Material color scheme: the neon/metal
 * glow color, card rim lighting, and the ambient background style. Keeping
 * these here means gauges, cards, nav and the ambient layer all restyle from a
 * single source when the theme changes.
 */
data class NovaThemeSpec(
    /** Primary glow color for halos, gauge rings and the logo bloom. */
    val glow: Color,
    /** Brighter core used at the center of glows / shimmer sweeps. */
    val glowBright: Color,
    /** Card rim (border) color — a thin metallic/neon edge on surfaces. */
    val rim: Color,
    /** Ambient background treatment. */
    val ambient: AmbientStyle,
    /** True when the background is light (Heritage) — flips scrim directions. */
    val lightBackground: Boolean = false
)

fun NovaTheme.gaugePalette(): GaugePalette = when (this) {
    NovaTheme.DARK_LUXE -> GaugePalette(
        arcStart = Color(0xFFD4AF37), arcEnd = Color(0xFF8B7500),
        buy = Color(0xFFE7C65C), sell = Color(0xFFB65D4A),
        hold = Color(0xFFAAA07A), needle = Color(0xFFF4D878),
        label = Color(0xFFE7C65C), glow = Color(0xFFD4AF37),
        background = Color(0xFF050505), motion = GaugeMotionStyle.CONFIDENT_SWEEP
    )
    NovaTheme.MINT_LUXE -> GaugePalette(
        arcStart = Color(0xFF00FFC6), arcEnd = Color(0xFF007F6E),
        buy = Color(0xFF5DFFE0), sell = Color(0xFFE27B83),
        hold = Color(0xFF9ABDB5), needle = Color(0xFFBFFFF1),
        label = Color(0xFFBFFFF1), glow = Color(0xFF00FFC6),
        background = Color(0xFF020A09), motion = GaugeMotionStyle.FLUID_WAVE
    )
    NovaTheme.AURORA_FLUX -> GaugePalette(
        arcStart = Color(0xFFE6A8A8), arcEnd = Color(0xFFB76E79),
        buy = Color(0xFFFFC0C8), sell = Color(0xFFD66E7B),
        hold = Color(0xFFC8A8AC), needle = Color(0xFFFFD8D8),
        label = Color(0xFFFFE4E0), glow = Color(0xFFE6A8A8),
        background = Color(0xFF0D0709), motion = GaugeMotionStyle.ELEGANT_SWIRL
    )
    NovaTheme.CRIMSON_PULSE -> GaugePalette(
        arcStart = Color(0xFFC5B358), arcEnd = Color(0xFFA67C52),
        buy = Color(0xFF769D70), sell = Color(0xFFC25C5C),
        hold = Color(0xFF9D9278), needle = Color(0xFFE2C879),
        label = Color(0xFFFFF1D2), glow = Color(0xFFC5B358),
        background = Color(0xFF8F8068), motion = GaugeMotionStyle.REFINED_GLIDE
    )
}

fun NovaTheme.spec(): NovaThemeSpec = when (this) {
    NovaTheme.DARK_LUXE -> NovaThemeSpec(
        glow = NovaGold,
        glowBright = NovaGoldBright,
        rim = NovaGold.copy(alpha = 0.55f),
        ambient = AmbientStyle.EMBER_GLOW
    )
    NovaTheme.MINT_LUXE -> NovaThemeSpec(
        glow = NovaMint,
        glowBright = Color(0xFFDFFFF2),
        rim = NovaMint.copy(alpha = 0.45f),
        ambient = AmbientStyle.AURORA_WISPS
    )
    NovaTheme.AURORA_FLUX -> NovaThemeSpec(
        glow = NovaRoseGlow,
        glowBright = Color(0xFFFFD9E0),
        rim = NovaRose.copy(alpha = 0.5f),
        ambient = AmbientStyle.LIGHT_RIBBONS
    )
    NovaTheme.CRIMSON_PULSE -> NovaThemeSpec(
        glow = NovaCopperBright,
        glowBright = Color(0xFFFFE9CF),
        rim = NovaCopper.copy(alpha = 0.7f),
        ambient = AmbientStyle.HERITAGE_PATTERN,
        lightBackground = true
    )
}
