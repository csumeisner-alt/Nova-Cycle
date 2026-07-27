package com.novacycle.ui.theme

import androidx.compose.ui.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the premium theme system invariants:
 *  - storage keys stay stable across the visual rebrand (persisted selections
 *    and unlock flags must keep resolving),
 *  - every theme has a distinct ambient treatment,
 *  - unlock thresholds are unchanged (10k / 20k),
 *  - Heritage is the only light-background theme.
 */
class NovaThemeSpecTest {

    @Test
    fun `storage keys are unchanged after rebrand`() {
        assertEquals("dark_luxe", NovaTheme.DARK_LUXE.storageKey)
        assertEquals("mint_luxe", NovaTheme.MINT_LUXE.storageKey)
        assertEquals("aurora_flux", NovaTheme.AURORA_FLUX.storageKey)
        assertEquals("crimson_pulse", NovaTheme.CRIMSON_PULSE.storageKey)
        // Old persisted keys still resolve to the rebranded themes
        assertEquals(NovaTheme.AURORA_FLUX, NovaTheme.fromStorageKey("aurora_flux"))
        assertEquals(NovaTheme.CRIMSON_PULSE, NovaTheme.fromStorageKey("crimson_pulse"))
    }

    @Test
    fun `unlock thresholds are unchanged`() {
        assertEquals(0L, NovaTheme.DARK_LUXE.unlockTaps)
        assertEquals(0L, NovaTheme.MINT_LUXE.unlockTaps)
        assertEquals(10_000L, NovaTheme.AURORA_FLUX.unlockTaps)
        assertEquals(20_000L, NovaTheme.CRIMSON_PULSE.unlockTaps)
    }

    @Test
    fun `every theme has a distinct ambient style`() {
        val styles = NovaTheme.entries.map { it.spec().ambient }
        assertEquals(styles.size, styles.toSet().size)
    }

    @Test
    fun `heritage is the only light-background theme`() {
        NovaTheme.entries.forEach { theme ->
            val light = theme.spec().lightBackground
            if (theme == NovaTheme.CRIMSON_PULSE) assertTrue(light)
            else assertTrue(!light)
        }
    }

    @Test
    fun `display names reflect the premium rebrand`() {
        assertEquals("Executive Gold", NovaTheme.DARK_LUXE.displayName)
        assertEquals("Mint Luxe", NovaTheme.MINT_LUXE.displayName)
        assertEquals("Rose Luxe", NovaTheme.AURORA_FLUX.displayName)
        assertEquals("Heritage", NovaTheme.CRIMSON_PULSE.displayName)
        assertNotEquals(NovaTheme.DARK_LUXE.spec().glow, NovaTheme.AURORA_FLUX.spec().glow)
    }

    @Test
    fun `each theme has a distinct gauge palette and motion style`() {
        val palettes = NovaTheme.entries.map { it.gaugePalette() }
        assertEquals(palettes.size, palettes.map { it.motion }.toSet().size)
        assertEquals(Color(0xFFD4AF37), NovaTheme.DARK_LUXE.gaugePalette().arcStart)
        assertEquals(Color(0xFF8B7500), NovaTheme.DARK_LUXE.gaugePalette().arcEnd)
        assertEquals(Color(0xFF00FFC6), NovaTheme.MINT_LUXE.gaugePalette().arcStart)
        assertEquals(Color(0xFF007F6E), NovaTheme.MINT_LUXE.gaugePalette().arcEnd)
        assertEquals(Color(0xFFE6A8A8), NovaTheme.AURORA_FLUX.gaugePalette().arcStart)
        assertEquals(Color(0xFFB76E79), NovaTheme.AURORA_FLUX.gaugePalette().arcEnd)
        assertEquals(Color(0xFFC5B358), NovaTheme.CRIMSON_PULSE.gaugePalette().arcStart)
        assertEquals(Color(0xFFA67C52), NovaTheme.CRIMSON_PULSE.gaugePalette().arcEnd)
    }
}
