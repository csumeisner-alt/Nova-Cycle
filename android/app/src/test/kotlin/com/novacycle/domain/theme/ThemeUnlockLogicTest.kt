package com.novacycle.domain.theme

import com.novacycle.ui.theme.AppTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ThemeUnlockLogicTest {

    // ── Achievement threshold ────────────────────────────────────────────

    @Test
    fun `unlock fires exactly on the 20000th tap`() {
        assertFalse(ThemeUnlockLogic.isUnlockTap(19_999))
        assertTrue(ThemeUnlockLogic.isUnlockTap(20_000))
        assertFalse(ThemeUnlockLogic.isUnlockTap(20_001))
    }

    @Test
    fun `achievementReached is monotonic at and beyond threshold`() {
        assertFalse(ThemeUnlockLogic.achievementReached(0))
        assertFalse(ThemeUnlockLogic.achievementReached(19_999))
        assertTrue(ThemeUnlockLogic.achievementReached(20_000))
        assertTrue(ThemeUnlockLogic.achievementReached(1_000_000))
    }

    // ── Availability rules ───────────────────────────────────────────────

    @Test
    fun `dark luxe is always available`() {
        assertTrue(ThemeUnlockLogic.isThemeAvailable(AppTheme.DARK_LUXE, false, false, false))
    }

    @Test
    fun `aurora and crimson require their unlock flags`() {
        assertFalse(ThemeUnlockLogic.isThemeAvailable(AppTheme.AURORA_FLUX, false, true, true))
        assertTrue(ThemeUnlockLogic.isThemeAvailable(AppTheme.AURORA_FLUX, true, false, false))
        assertFalse(ThemeUnlockLogic.isThemeAvailable(AppTheme.CRIMSON_PULSE, true, false, true))
        assertTrue(ThemeUnlockLogic.isThemeAvailable(AppTheme.CRIMSON_PULSE, false, true, false))
    }

    @Test
    fun `mint luxe requires purchase flag only`() {
        assertFalse(ThemeUnlockLogic.isThemeAvailable(AppTheme.MINT_LUXE, true, true, false))
        assertTrue(ThemeUnlockLogic.isThemeAvailable(AppTheme.MINT_LUXE, false, false, true))
    }

    // ── Selection sanitization ───────────────────────────────────────────

    @Test
    fun `locked selection falls back to dark luxe`() {
        assertEquals(
            AppTheme.DARK_LUXE,
            ThemeUnlockLogic.sanitizeSelection(AppTheme.MINT_LUXE, true, true, false)
        )
        assertEquals(
            AppTheme.DARK_LUXE,
            ThemeUnlockLogic.sanitizeSelection(AppTheme.AURORA_FLUX, false, false, false)
        )
    }

    @Test
    fun `unlocked selection is preserved`() {
        assertEquals(
            AppTheme.MINT_LUXE,
            ThemeUnlockLogic.sanitizeSelection(AppTheme.MINT_LUXE, false, false, true)
        )
        assertEquals(
            AppTheme.CRIMSON_PULSE,
            ThemeUnlockLogic.sanitizeSelection(AppTheme.CRIMSON_PULSE, true, true, false)
        )
    }

    // ── Storage key round-trip ───────────────────────────────────────────

    @Test
    fun `storage keys round-trip and unknown keys default to dark luxe`() {
        AppTheme.entries.forEach { theme ->
            assertEquals(theme, AppTheme.fromStorageKey(theme.storageKey))
        }
        assertEquals(AppTheme.DARK_LUXE, AppTheme.fromStorageKey(null))
        assertEquals(AppTheme.DARK_LUXE, AppTheme.fromStorageKey("garbage"))
    }

    // ── Progress display ─────────────────────────────────────────────────

    @Test
    fun `progress label formats with separators and caps at threshold`() {
        assertEquals("12,345 / 20,000 taps", ThemeUnlockLogic.progressLabel(12_345))
        assertEquals("20,000 / 20,000 taps", ThemeUnlockLogic.progressLabel(25_000))
        assertEquals("0 / 20,000 taps", ThemeUnlockLogic.progressLabel(0))
    }

    @Test
    fun `progress fraction is clamped to unit interval`() {
        assertEquals(0f, ThemeUnlockLogic.progressFraction(0))
        assertEquals(0.5f, ThemeUnlockLogic.progressFraction(10_000), 1e-6f)
        assertEquals(1f, ThemeUnlockLogic.progressFraction(40_000))
    }
}
