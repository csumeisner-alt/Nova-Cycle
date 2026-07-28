package com.novacycle.ui.components.confidence

import com.novacycle.domain.model.ConfidencePoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ConfidenceChartHelpersTest {

    private fun pt(ts: String, long: Float, short: Float) = ConfidencePoint(
        timestamp = ts,
        longBuyConfidence = long, longSellConfidence = 0f,
        shortBuyConfidence = short, shortSellConfidence = 0f
    )

    @Test
    fun `crossover found at last crossing`() {
        val points = listOf(
            pt("2026-07-27T10:00:00", 40f, 60f),
            pt("2026-07-27T11:00:00", 55f, 50f),  // cross 1
            pt("2026-07-27T12:00:00", 60f, 45f),
            pt("2026-07-27T13:00:00", 44f, 52f)   // cross 2 (last)
        )
        assertEquals(3, lastCrossoverIndex(points))
    }

    @Test
    fun `no crossover returns null`() {
        val points = listOf(
            pt("2026-07-27T10:00:00", 70f, 30f),
            pt("2026-07-27T11:00:00", 65f, 35f),
            pt("2026-07-27T12:00:00", 72f, 40f)
        )
        assertNull(lastCrossoverIndex(points))
    }

    @Test
    fun `hour windows use time-only labels`() {
        assertEquals("14:30", formatTimeLabel("2026-07-27T14:30:00", "3h"))
        assertEquals("14:30", formatTimeLabel("2026-07-27T14:30:00", "12h"))
    }

    @Test
    fun `month windows use date labels`() {
        assertEquals("Jul 27", formatTimeLabel("2026-07-27T14:30:00", "3mo"))
        assertEquals("Jul 27", formatTimeLabel("2026-07-27T14:30:00", "30d"))
    }

    @Test
    fun `unparseable timestamp falls back to raw prefix`() {
        // fallback is first 10 chars of the raw string
        assertEquals("not-a-time", formatTimeLabel("not-a-timestamp", "24h"))
    }

    @Test
    fun `tooltip timestamp is readable`() {
        assertEquals("Jul 27, 2026  14:30", formatTooltipTimestamp("2026-07-27T14:30:00"))
    }
}
