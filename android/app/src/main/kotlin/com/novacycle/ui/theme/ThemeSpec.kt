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

enum class GaugeNeedleTip { SHARP, GLOW_DOT, GLOW_HALO, BLUNT }
enum class GaugeHubStyle { GOLD_RING, TEAL_DOT, ROSE_GLOW, COPPER_RING }

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
    val lightBackground: Boolean = false,
    
    val gaugeArcStroke: Float = 14f,
    val gaugeTickCount: Int = 5,
    val gaugeTickLong: Boolean = false,
    val gaugeNeedleTip: GaugeNeedleTip = GaugeNeedleTip.SHARP,
    val gaugeHubStyle: GaugeHubStyle = GaugeHubStyle.GOLD_RING,
    val gaugeGlowIntensity: Float = 0.18f,
    val gaugeShowTicks: Boolean = true
)

fun NovaTheme.spec(): NovaThemeSpec = when (this) {
    NovaTheme.DARK_LUXE -> NovaThemeSpec(
        glow = NovaGold,
        glowBright = NovaGoldBright,
        rim = NovaGold.copy(alpha = 0.55f),
        ambient = AmbientStyle.EMBER_GLOW,
        gaugeArcStroke = 14f,
        gaugeTickCount = 5,
        gaugeTickLong = true,
        gaugeNeedleTip = GaugeNeedleTip.SHARP,
        gaugeHubStyle = GaugeHubStyle.GOLD_RING,
        gaugeGlowIntensity = 0.18f,
        gaugeShowTicks = true
    )
    NovaTheme.MINT_LUXE -> NovaThemeSpec(
        glow = NovaMint,
        glowBright = Color(0xFFDFFFF2),
        rim = NovaMint.copy(alpha = 0.45f),
        ambient = AmbientStyle.AURORA_WISPS,
        gaugeArcStroke = 22f,
        gaugeTickCount = 3,
        gaugeTickLong = false,
        gaugeNeedleTip = GaugeNeedleTip.GLOW_DOT,
        gaugeHubStyle = GaugeHubStyle.TEAL_DOT,
        gaugeGlowIntensity = 0.25f,
        gaugeShowTicks = true
    )
    NovaTheme.AURORA_FLUX -> NovaThemeSpec(
        glow = NovaRoseGlow,
        glowBright = Color(0xFFFFD9E0),
        rim = NovaRose.copy(alpha = 0.5f),
        ambient = AmbientStyle.LIGHT_RIBBONS,
        gaugeArcStroke = 28f,
        gaugeTickCount = 0,
        gaugeTickLong = false,
        gaugeNeedleTip = GaugeNeedleTip.GLOW_HALO,
        gaugeHubStyle = GaugeHubStyle.ROSE_GLOW,
        gaugeGlowIntensity = 0.45f,
        gaugeShowTicks = false
    )
    NovaTheme.CRIMSON_PULSE -> NovaThemeSpec(
        glow = NovaCopperBright,
        glowBright = Color(0xFFFFE9CF),
        rim = NovaCopper.copy(alpha = 0.7f),
        ambient = AmbientStyle.HERITAGE_PATTERN,
        lightBackground = true,
        gaugeArcStroke = 20f,
        gaugeTickCount = 5,
        gaugeTickLong = true,
        gaugeNeedleTip = GaugeNeedleTip.BLUNT,
        gaugeHubStyle = GaugeHubStyle.COPPER_RING,
        gaugeGlowIntensity = 0.12f,
        gaugeShowTicks = true
    )
}
