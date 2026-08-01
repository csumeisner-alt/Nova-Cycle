package com.novacycle.ui.components

import com.novacycle.data.remote.models.CandleResponse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the pure chart math in ChartFrame.kt:
 * [candleIndexAt] (touch x → candle index under pan/zoom),
 * [niceTicks] (price axis tick generation) and
 * [signalIndexByCandle] (signal timestamp → candle bucket mapping).
 */
class ChartFrameHelpersTest {

    // ---------- candleIndexAt ----------

    private val padding = 16f

    @Test
    fun `no pan no zoom maps touch to the bar under the finger`() {
        // barWidth 10, offset 0: candle i occupies [padding + 10i, padding + 10(i+1))
        assertEquals(0, candleIndexAt(padding + 0f, 10f, 0f, padding, 100))
        assertEquals(0, candleIndexAt(padding + 9.9f, 10f, 0f, padding, 100))
        assertEquals(1, candleIndexAt(padding + 10f, 10f, 0f, padding, 100))
        assertEquals(37, candleIndexAt(padding + 375f, 10f, 0f, padding, 100))
    }

    @Test
    fun `touch center of each bar returns that bar for every index`() {
        val barWidth = 7.3f
        val offsetX = -123.4f
        for (i in 0 until 200) {
            val centerX = i * barWidth + offsetX + padding + barWidth / 2
            assertEquals("candle $i", i, candleIndexAt(centerX, barWidth, offsetX, padding, 200))
        }
    }

    @Test
    fun `max zoom huge bar width still resolves correct index`() {
        // Extreme zoom-in: bars 400px wide, panned far left.
        val barWidth = 400f
        val offsetX = -400f * 95
        val centerOf97 = 97 * barWidth + offsetX + padding + barWidth / 2
        assertEquals(97, candleIndexAt(centerOf97, barWidth, offsetX, padding, 100))
    }

    @Test
    fun `far pan right of the last candle returns null`() {
        // 50 candles, barWidth 10 → series ends at offset+padding+500
        assertNull(candleIndexAt(padding + 500f, 10f, 0f, padding, 50))
        assertNull(candleIndexAt(padding + 10_000f, 10f, 0f, padding, 50))
    }

    @Test
    fun `touch left of the first candle returns null`() {
        assertNull(candleIndexAt(padding - 0.1f, 10f, 0f, padding, 50))
        assertNull(candleIndexAt(0f, 10f, 0f, padding, 50))
        // Panned right so the series starts mid-plot; touching before it is null.
        assertNull(candleIndexAt(padding + 50f, 10f, 100f, padding, 50))
    }

    @Test
    fun `pan offset shifts the mapping consistently`() {
        val barWidth = 12f
        // Panned left by exactly 5 bars: touching x that showed candle 0 now shows candle 5.
        val offsetX = -5 * barWidth
        assertEquals(5, candleIndexAt(padding + barWidth / 2, barWidth, offsetX, padding, 100))
        // Panned right: the first visible bar is candle 0 shifted right.
        assertEquals(0, candleIndexAt(padding + 3 * barWidth + barWidth / 2, barWidth, 3 * barWidth, padding, 100))
    }

    @Test
    fun `zero candles always returns null`() {
        assertNull(candleIndexAt(padding + 5f, 10f, 0f, padding, 0))
    }

    @Test
    fun `tiny bar width from extreme zoom out still lands on distinct candles`() {
        val barWidth = 0.5f
        assertEquals(0, candleIndexAt(padding + 0.25f, barWidth, 0f, padding, 1000))
        assertEquals(999, candleIndexAt(padding + 999 * barWidth + 0.25f, barWidth, 0f, padding, 1000))
        assertNull(candleIndexAt(padding + 1000 * barWidth, barWidth, 0f, padding, 1000))
    }

    // ---------- niceTicks ----------

    @Test
    fun `ticks are evenly spaced with a 1-2-5 step and cover the range`() {
        val ticks = niceTicks(680f, 690f)
        assertTrue(ticks.isNotEmpty())
        val step = ticks[1] - ticks[0]
        // step should be one of 1/2/5 × 10^n
        val mantissa = step / Math.pow(10.0, Math.floor(Math.log10(step.toDouble()))).toFloat()
        assertTrue("mantissa=$mantissa", listOf(1f, 2f, 5f).any { Math.abs(it - mantissa) < 1e-3 })
        ticks.zipWithNext { a, b -> assertEquals(step, b - a, step * 1e-3f) }
        assertTrue(ticks.first() >= 680f)
        assertTrue(ticks.last() <= 690f + step * 0.001f)
    }

    @Test
    fun `degenerate ranges return empty`() {
        assertTrue(niceTicks(100f, 100f).isEmpty())
        assertTrue(niceTicks(100f, 99f).isEmpty())
        assertTrue(niceTicks(0f, 10f, targetCount = 1).isEmpty())
    }

    @Test
    fun `tick values land on round numbers`() {
        val ticks = niceTicks(682.13f, 687.91f)
        // Each tick must be a multiple of the step (round values like 683.0, 684.0…)
        val step = ticks[1] - ticks[0]
        ticks.forEach { t ->
            val ratio = t / step
            assertEquals("tick $t not multiple of $step", Math.round(ratio).toFloat(), ratio, 1e-3f)
        }
    }

    @Test
    fun `small fractional range produces sub-dollar ticks`() {
        val ticks = niceTicks(100.00f, 100.10f)
        assertTrue(ticks.size >= 2)
        assertTrue(ticks[1] - ticks[0] <= 0.05f + 1e-6f)
    }

    // ---------- signalIndexByCandle ----------

    private fun candle(ts: String) = CandleResponse(
        timestamp = ts, open = 1f, high = 2f, low = 0.5f, close = 1.5f
    )

    private val candles = listOf(
        candle("2026-07-31T13:30:00"),
        candle("2026-07-31T13:35:00"),
        candle("2026-07-31T13:40:00"),
        candle("2026-07-31T13:45:00")
    )

    @Test
    fun `exact timestamp match maps to that candle`() {
        val out = signalIndexByCandle(candles, listOf("2026-07-31T13:40:00"))
        assertEquals(2, out["2026-07-31T13:40:00"])
    }

    @Test
    fun `signal inside a bucket maps to the bucket start candle`() {
        val out = signalIndexByCandle(candles, listOf("2026-07-31T13:37:12"))
        assertEquals(1, out["2026-07-31T13:37:12"])
    }

    @Test
    fun `signal before the first candle is dropped`() {
        val out = signalIndexByCandle(candles, listOf("2026-07-31T13:00:00"))
        assertTrue(out.isEmpty())
    }

    @Test
    fun `signal after the last candle maps to the last candle`() {
        val out = signalIndexByCandle(candles, listOf("2026-07-31T15:59:00"))
        assertEquals(3, out["2026-07-31T15:59:00"])
    }

    @Test
    fun `malformed signal timestamp is skipped without throwing`() {
        val out = signalIndexByCandle(candles, listOf("not-a-timestamp", "2026-07-31T13:35:00"))
        assertEquals(1, out.size)
        assertEquals(1, out["2026-07-31T13:35:00"])
    }

    @Test
    fun `empty candle list yields empty map`() {
        assertTrue(signalIndexByCandle(emptyList(), listOf("2026-07-31T13:35:00")).isEmpty())
    }

    @Test
    fun `multiple signals map independently`() {
        val signals = listOf("2026-07-31T13:30:00", "2026-07-31T13:44:59", "2026-07-31T13:45:00")
        val out = signalIndexByCandle(candles, signals)
        assertEquals(0, out["2026-07-31T13:30:00"])
        assertEquals(2, out["2026-07-31T13:44:59"])
        assertEquals(3, out["2026-07-31T13:45:00"])
    }
}
