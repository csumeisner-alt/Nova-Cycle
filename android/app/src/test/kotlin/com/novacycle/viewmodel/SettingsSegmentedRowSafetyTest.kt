package com.novacycle.viewmodel

import com.novacycle.domain.model.NotifSensitivity
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SmoothingMode
import com.novacycle.domain.model.StoryLevel
import com.novacycle.domain.model.WeightingMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the enum ordinal assumptions that [SettingsScreen] passes to
 * [SegmentedRow] via `settings.<enum>.ordinal`.
 *
 * SegmentedRow renders `options[selectedIndex]` where the options list is
 * generated from `<Enum>.entries`. If an enum value is added, removed, or
 * reordered without updating the when-expression that builds the options list,
 * the ordinal and the list length can diverge and cause an
 * IndexOutOfBoundsException during composition (which shows as a crash when
 * the Settings tab is opened).
 *
 * These tests pin the exact ordinal ↔ label mappings that the screen relies on
 * so any future enum change that breaks Settings fails here first.
 */
class SettingsSegmentedRowSafetyTest {

    // ── WeightingMode ──────────────────────────────────────────────────────────

    @Test
    fun `WeightingMode entries count matches SettingsScreen options list`() {
        // SettingsScreen maps WeightingMode.entries to ["Balanced", "Indicator", "ML-Heavy"]
        assertEquals("WeightingMode entry count must equal the SegmentedRow options list size",
            3, WeightingMode.entries.size)
    }

    @Test
    fun `WeightingMode ordinals are stable and in-bounds`() {
        val optionCount = WeightingMode.entries.size
        WeightingMode.entries.forEach { mode ->
            assertTrue(
                "WeightingMode.${mode.name}.ordinal=${mode.ordinal} is out of range [0,$optionCount)",
                mode.ordinal in 0 until optionCount
            )
        }
        assertEquals(0, WeightingMode.BALANCED.ordinal)
        assertEquals(1, WeightingMode.INDICATOR_HEAVY.ordinal)
        assertEquals(2, WeightingMode.ML_HEAVY.ordinal)
    }

    // ── SmoothingMode ──────────────────────────────────────────────────────────

    @Test
    fun `SmoothingMode entries count matches SettingsScreen options list`() {
        // SettingsScreen maps SmoothingMode.entries to ["Raw", "Light", "EMA", "Heavy"]
        assertEquals("SmoothingMode entry count must equal the SegmentedRow options list size",
            4, SmoothingMode.entries.size)
    }

    @Test
    fun `SmoothingMode ordinals are stable and in-bounds`() {
        val optionCount = SmoothingMode.entries.size
        SmoothingMode.entries.forEach { mode ->
            assertTrue(
                "SmoothingMode.${mode.name}.ordinal=${mode.ordinal} is out of range [0,$optionCount)",
                mode.ordinal in 0 until optionCount
            )
        }
        assertEquals(0, SmoothingMode.RAW.ordinal)
        assertEquals(1, SmoothingMode.LIGHT.ordinal)
        assertEquals(2, SmoothingMode.EMA.ordinal)
        assertEquals(3, SmoothingMode.HEAVY.ordinal)
    }

    // ── StoryLevel ─────────────────────────────────────────────────────────────

    @Test
    fun `StoryLevel entries count matches SettingsScreen options list`() {
        // SettingsScreen maps StoryLevel.entries to ["Simple", "Advanced", "Expert"]
        assertEquals("StoryLevel entry count must equal the SegmentedRow options list size",
            3, StoryLevel.entries.size)
    }

    @Test
    fun `StoryLevel ordinals are stable and in-bounds`() {
        val optionCount = StoryLevel.entries.size
        StoryLevel.entries.forEach { level ->
            assertTrue(
                "StoryLevel.${level.name}.ordinal=${level.ordinal} is out of range [0,$optionCount)",
                level.ordinal in 0 until optionCount
            )
        }
        assertEquals(0, StoryLevel.SIMPLE.ordinal)
        assertEquals(1, StoryLevel.ADVANCED.ordinal)
        assertEquals(2, StoryLevel.EXPERT.ordinal)
    }

    // ── NotifSensitivity ───────────────────────────────────────────────────────

    @Test
    fun `NotifSensitivity entries are all handled in SettingsScreen when-expression`() {
        // SettingsScreen has a when-expression over NotifSensitivity.entries with three branches.
        // If a new value is added without updating the when-expression, it will crash with a
        // non-exhaustive when on an unknown branch (Kotlin exhaustive when is compile-time only
        // for sealed classes; enums throw at runtime if a branch is missing).
        val expectedEntries = setOf(
            NotifSensitivity.STANDARD,
            NotifSensitivity.HIGH,
            NotifSensitivity.LOW
        )
        assertEquals(
            "NotifSensitivity has a new entry — update the when-expression in SettingsScreen",
            expectedEntries, NotifSensitivity.entries.toSet()
        )
    }

    // ── SensitivitySettings defaults ───────────────────────────────────────────

    @Test
    fun `SensitivitySettings default enum values have valid ordinals for their options lists`() {
        val defaults = SensitivitySettings()
        // WeightingMode.BALANCED.ordinal must be in [0, WeightingMode.entries.size)
        assertTrue(defaults.weightingMode.ordinal in WeightingMode.entries.indices)
        // SmoothingMode.RAW.ordinal must be in [0, SmoothingMode.entries.size)
        assertTrue(defaults.smoothingMode.ordinal in SmoothingMode.entries.indices)
        // StoryLevel.SIMPLE.ordinal must be in [0, StoryLevel.entries.size)
        assertTrue(defaults.storyCardLevel.ordinal in StoryLevel.entries.indices)
    }
}
