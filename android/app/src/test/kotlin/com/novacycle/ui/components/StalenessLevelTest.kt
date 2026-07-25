package com.novacycle.ui.components

import org.junit.Assert.assertEquals
import org.junit.Test

class StalenessLevelTest {

    private val now = 1_000_000_000_000L
    private fun ageMin(min: Long) = now - min * 60_000L

    @Test
    fun `fresh below warning threshold`() {
        assertEquals(StalenessLevel.FRESH, stalenessLevel(now, now))
        assertEquals(StalenessLevel.FRESH, stalenessLevel(now, ageMin(4)))
    }

    @Test
    fun `warning at and past warning threshold`() {
        assertEquals(StalenessLevel.WARNING, stalenessLevel(now, ageMin(5)))
        assertEquals(StalenessLevel.WARNING, stalenessLevel(now, ageMin(14)))
    }

    @Test
    fun `critical at and past critical threshold`() {
        assertEquals(StalenessLevel.CRITICAL, stalenessLevel(now, ageMin(15)))
        assertEquals(StalenessLevel.CRITICAL, stalenessLevel(now, ageMin(120)))
    }

    @Test
    fun `custom thresholds are respected`() {
        val warn = 60_000L
        val crit = 120_000L
        assertEquals(StalenessLevel.FRESH, stalenessLevel(now, now - 59_000L, warn, crit))
        assertEquals(StalenessLevel.WARNING, stalenessLevel(now, now - 60_000L, warn, crit))
        assertEquals(StalenessLevel.CRITICAL, stalenessLevel(now, now - 120_000L, warn, crit))
    }

    @Test
    fun `future timestamps clamp to fresh`() {
        assertEquals(StalenessLevel.FRESH, stalenessLevel(now, now + 60_000L))
    }
}
