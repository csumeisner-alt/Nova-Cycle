package com.novacycle.ui.screens

import org.junit.Assert.assertEquals
import org.junit.Test

class SettingsThemeUnlockLabelTest {

    @Test
    fun `locked theme progress label renders percent and grouped tap count`() {
        assertEquals(
            "50% · 5,000 taps to unlock",
            formatThemeUnlockProgress(percent = 50, remainingTaps = 5_000L)
        )
    }

    @Test
    fun `locked theme progress label handles zero remaining taps`() {
        assertEquals(
            "100% · 0 taps to unlock",
            formatThemeUnlockProgress(percent = 100, remainingTaps = 0L)
        )
    }
}