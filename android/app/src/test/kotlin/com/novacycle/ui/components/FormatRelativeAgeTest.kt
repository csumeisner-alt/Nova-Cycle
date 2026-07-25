package com.novacycle.ui.components

import org.junit.Assert.assertEquals
import org.junit.Test

class FormatRelativeAgeTest {

    private val now = 1_000_000_000_000L
    private fun ageSec(sec: Long) = now - sec * 1000L
    private fun ageMin(min: Long) = now - min * 60_000L
    private fun ageHours(h: Long, remMin: Long = 0L) = now - (h * 60 + remMin) * 60_000L
    private fun ageDays(d: Long) = now - d * 24 * 60 * 60_000L

    @Test
    fun `just now for ages under one minute`() {
        assertEquals("just now", formatRelativeAge(now, now))
        assertEquals("just now", formatRelativeAge(now, ageSec(1)))
        assertEquals("just now", formatRelativeAge(now, ageSec(59)))
    }

    @Test
    fun `boundary 59s to 1 min`() {
        assertEquals("just now", formatRelativeAge(now, ageSec(59)))
        assertEquals("1 min ago", formatRelativeAge(now, ageSec(60)))
    }

    @Test
    fun `minutes between 1 and 59`() {
        assertEquals("1 min ago", formatRelativeAge(now, ageMin(1)))
        assertEquals("12 min ago", formatRelativeAge(now, ageMin(12)))
        assertEquals("59 min ago", formatRelativeAge(now, ageMin(59)))
    }

    @Test
    fun `boundary 59 min to 1 h`() {
        assertEquals("59 min ago", formatRelativeAge(now, ageMin(59)))
        assertEquals("1 h ago", formatRelativeAge(now, ageMin(60)))
    }

    @Test
    fun `hours without remainder minutes`() {
        assertEquals("1 h ago", formatRelativeAge(now, ageHours(1)))
        assertEquals("3 h ago", formatRelativeAge(now, ageHours(3)))
        assertEquals("23 h ago", formatRelativeAge(now, ageHours(23)))
    }

    @Test
    fun `hours with remainder minutes`() {
        assertEquals("1 h 1 min ago", formatRelativeAge(now, ageHours(1, 1)))
        assertEquals("3 h 5 min ago", formatRelativeAge(now, ageHours(3, 5)))
        assertEquals("23 h 59 min ago", formatRelativeAge(now, ageHours(23, 59)))
    }

    @Test
    fun `boundary 23 h to 1 day`() {
        assertEquals("23 h 59 min ago", formatRelativeAge(now, ageHours(23, 59)))
        assertEquals("1 day ago", formatRelativeAge(now, ageHours(24)))
    }

    @Test
    fun `singular one day`() {
        assertEquals("1 day ago", formatRelativeAge(now, ageDays(1)))
        // Still 1 day until 48 h.
        assertEquals("1 day ago", formatRelativeAge(now, ageHours(47, 59)))
    }

    @Test
    fun `plural days`() {
        assertEquals("2 days ago", formatRelativeAge(now, ageDays(2)))
        assertEquals("30 days ago", formatRelativeAge(now, ageDays(30)))
    }

    @Test
    fun `future timestamps clamp to just now`() {
        assertEquals("just now", formatRelativeAge(now, now + 1000L))
        assertEquals("just now", formatRelativeAge(now, now + 24 * 60 * 60_000L))
    }

    @Test
    fun `sub-second differences truncate to just now`() {
        assertEquals("just now", formatRelativeAge(now, now - 999L))
    }
}
