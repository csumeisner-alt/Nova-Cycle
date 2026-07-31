package com.novacycle.ui.screens

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the mathematical core of the AccuracySparkline composable.
 *
 * AccuracySparkline (ReliabilityScreen.kt) computes y-positions with:
 *
 *   val range = (max - min).takeIf { it > 1e-6f } ?: 1f
 *   fun yFor(v: Float) = pad + (1f - (v - min) / range) * usable
 *
 * When all values are identical the raw range is 0f, which would produce
 * NaN offsets (division by zero) without the guard. These tests verify:
 *   1. The guard substitutes 1f for a zero range so yFor() stays finite.
 *   2. A flat trend places all points at the same y (mid-height after padding).
 *   3. A normal trend with spread produces finite, bounded y-values.
 *   4. A two-value flat list (minimum list that reaches the drawing loop) is safe.
 *   5. A single-value list never reaches the drawing loop (size < 2 short-circuits).
 */
class AccuracySparklineLogicTest {

    // ── Helpers mirroring the composable's logic exactly ─────────────────────

    /** Replicates `(max - min).takeIf { it > 1e-6f } ?: 1f` from the composable. */
    private fun safeRange(values: List<Float>): Float {
        val min = values.min()
        val max = values.max()
        return (max - min).takeIf { it > 1e-6f } ?: 1f
    }

    /**
     * Replicates the `yFor` lambda from the composable given explicit layout
     * dimensions. Uses the same pad = height * 0.1 and usable = height - 2*pad
     * so the computed offsets are comparable to what the canvas would draw.
     */
    private fun yFor(v: Float, min: Float, range: Float, height: Float): Float {
        val pad = height * 0.1f
        val usable = height - 2f * pad
        return pad + (1f - (v - min) / range) * usable
    }

    private fun yPositions(values: List<Float>, height: Float = 36f): List<Float> {
        val min = values.min()
        val range = safeRange(values)
        return values.map { yFor(it, min, range, height) }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    @Test
    fun `flat trend with 3 identical values uses safe range of 1f`() {
        val values = listOf(0.70f, 0.70f, 0.70f)
        val range = safeRange(values)
        // Raw range is 0; guard must substitute 1f.
        assertEquals(1f, range, 1e-7f)
    }

    @Test
    fun `flat trend y-positions are all finite and identical`() {
        val values = listOf(0.70f, 0.70f, 0.70f)
        val ys = yPositions(values)
        // No NaN or Infinity — the zero-range guard prevents division by zero.
        assertTrue("all y-values must be finite", ys.all { it.isFinite() })
        // All points land at the same y because all values are equal.
        val first = ys.first()
        assertTrue("all y-values must be identical for a flat trend",
            ys.all { kotlin.math.abs(it - first) < 1e-5f })
    }

    @Test
    fun `two-value flat list is safe and produces two identical y-positions`() {
        // Minimum list size that reaches the drawing loop (size >= 2).
        val values = listOf(0.60f, 0.60f)
        val range = safeRange(values)
        assertEquals("range guard must produce 1f for a 2-value flat list", 1f, range, 1e-7f)

        val ys = yPositions(values)
        assertEquals(2, ys.size)
        assertTrue("y-values must be finite", ys.all { it.isFinite() })
        assertEquals("both y-values must be equal", ys[0], ys[1], 1e-5f)
    }

    @Test
    fun `single value list is safe and guard range is still 1f`() {
        // The composable short-circuits at size < 2, so yFor is never called.
        // But the range computation itself must not crash or produce NaN.
        val values = listOf(0.55f)
        val range = safeRange(values)
        assertEquals("range guard must produce 1f for a single-value list", 1f, range, 1e-7f)
        // Verify yFor is also finite, even though the composable never calls it here.
        val y = yFor(values[0], values.min(), range, 36f)
        assertTrue("yFor must be finite for a single-value list", y.isFinite())
    }

    @Test
    fun `normal spread produces finite y-values bounded within canvas height`() {
        val values = listOf(0.50f, 0.55f, 0.62f, 0.58f, 0.65f)
        val height = 36f
        val ys = yPositions(values, height)
        assertTrue("all y-values must be finite", ys.all { it.isFinite() })
        // y must stay within [0, height] — pad shrinks the usable area, not expands it.
        assertTrue("all y-values must be within canvas height",
            ys.all { it >= 0f && it <= height })
    }

    @Test
    fun `range guard does not activate when spread exceeds 1e-6`() {
        // Two values that differ by more than 1e-6 must use the real range.
        val values = listOf(0.60f, 0.65f)
        val expectedRange = 0.65f - 0.60f
        val range = safeRange(values)
        assertEquals(expectedRange, range, 1e-7f)
        // Confirm the values are distinct (not collapsed to a flat line).
        val ys = yPositions(values)
        assertFalse("y-values must differ for a non-flat trend",
            kotlin.math.abs(ys[0] - ys[1]) < 1e-5f)
    }

    @Test
    fun `extreme flat trend at 0 percent accuracy is safe`() {
        val values = listOf(0f, 0f, 0f)
        val range = safeRange(values)
        assertEquals(1f, range, 1e-7f)
        val ys = yPositions(values)
        assertTrue("y-values must be finite even at 0% accuracy", ys.all { it.isFinite() })
    }

    @Test
    fun `extreme flat trend at 100 percent accuracy is safe`() {
        val values = listOf(1f, 1f, 1f)
        val range = safeRange(values)
        assertEquals(1f, range, 1e-7f)
        val ys = yPositions(values)
        assertTrue("y-values must be finite even at 100% accuracy", ys.all { it.isFinite() })
    }
}
