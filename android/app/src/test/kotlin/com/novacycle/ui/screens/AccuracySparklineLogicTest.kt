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
 *   val rawRange = max - min
 *   val range = rawRange.takeIf { it > 1e-6f } ?: 1f
 *   // When flat, shift effectiveMin so the line lands at mid-height (not bottom edge).
 *   val effectiveMin = if (rawRange > 1e-6f) min else min - 0.5f
 *   fun yFor(v: Float) = pad + (1f - (v - effectiveMin) / range) * usable
 *
 * When all values are identical the raw range is 0f, which would produce
 * NaN offsets (division by zero) without the guard. These tests verify:
 *   1. The guard substitutes 1f for a zero range so yFor() stays finite.
 *   2. A flat trend places all points at mid-height (pad + 0.5 * usable = height / 2).
 *   3. A normal trend with spread produces finite, bounded y-values.
 *   4. A two-value flat list (minimum list that reaches the drawing loop) is safe.
 *   5. A single-value list never reaches the drawing loop (size < 2 short-circuits).
 *   6. A near-flat trend (rawRange just above 1e-6f) uses the real range and maps
 *      the min–max spread to the FULL usable height (normalization is always full-span),
 *      so the sparkline is always visible regardless of how small the absolute spread is.
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
     * [effectiveMin] is the (possibly shifted) minimum used for normalisation.
     */
    private fun yFor(v: Float, effectiveMin: Float, range: Float, height: Float): Float {
        val pad = height * 0.1f
        val usable = height - 2f * pad
        return pad + (1f - (v - effectiveMin) / range) * usable
    }

    /**
     * Replicates the effectiveMin selection from the composable:
     * when the trend is flat, min is shifted down by 0.5 so the normalised
     * position is 0.5 and the line lands at mid-height instead of the bottom.
     */
    private fun effectiveMin(values: List<Float>): Float {
        val min = values.min()
        val max = values.max()
        val rawRange = max - min
        return if (rawRange > 1e-6f) min else min - 0.5f
    }

    private fun yPositions(values: List<Float>, height: Float = 36f): List<Float> {
        val effMin = effectiveMin(values)
        val range = safeRange(values)
        return values.map { yFor(it, effMin, range, height) }
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
    fun `flat trend y-position lands at mid-height not at the bottom edge`() {
        // With range = 0, the composable shifts effectiveMin by -0.5 so that
        // the normalised position of every point is 0.5. The expected y is:
        //   pad + (1 - 0.5) * usable  =  pad + usable/2  =  height/2
        val height = 36f
        val values = listOf(0.70f, 0.70f, 0.70f)
        val ys = yPositions(values, height)
        val midHeight = height / 2f
        ys.forEach { y ->
            assertEquals(
                "flat trend must land at mid-height (height/2), not at the bottom edge (height - pad)",
                midHeight, y, 1e-4f
            )
        }
        // Guard: bottom edge would be height - height*0.1 = height*0.9; confirm we are NOT there.
        val bottomEdge = height - height * 0.1f
        ys.forEach { y ->
            assertTrue(
                "flat trend must not render at the bottom edge ($bottomEdge); got $y",
                kotlin.math.abs(y - bottomEdge) > 1f
            )
        }
    }

    @Test
    fun `flat trend at 0 percent accuracy also lands at mid-height`() {
        val height = 36f
        val values = listOf(0f, 0f, 0f)
        val ys = yPositions(values, height)
        val midHeight = height / 2f
        ys.forEach { y ->
            assertEquals("0% flat trend must land at mid-height", midHeight, y, 1e-4f)
        }
    }

    @Test
    fun `flat trend at 100 percent accuracy also lands at mid-height`() {
        val height = 36f
        val values = listOf(1f, 1f, 1f)
        val ys = yPositions(values, height)
        val midHeight = height / 2f
        ys.forEach { y ->
            assertEquals("100% flat trend must land at mid-height", midHeight, y, 1e-4f)
        }
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
        val effMin = effectiveMin(values)
        val y = yFor(values[0], effMin, range, 36f)
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

    // ── Near-flat edge-case tests ─────────────────────────────────────────────
    //
    // When rawRange is just above 1e-6f (e.g. a tiny 0.00001 pt difference
    // between two retrains), the flat-trend guard does NOT activate and the real
    // range is used for normalisation.  Because the composable normalises v via
    //   (v - min) / range
    // the min point maps to 0 and the max point maps to 1 in normalised space,
    // so their y-positions are pad (top) and pad + usable (bottom) respectively —
    // i.e. the line always spans the FULL usable height regardless of how small
    // the absolute spread is.  This makes the sparkline inherently visible.
    //
    // The tests below verify that property for a variety of near-flat spreads
    // and confirm every y-position remains within the padded canvas area.

    @Test
    fun `near-flat spread of 1e-5 maps min and max to the full usable height`() {
        // rawRange = 1e-5f is just above the 1e-6f flat-trend threshold.
        // The guard must NOT activate, and the visual span (|yFor(min) - yFor(max)|)
        // must equal the full usable height so the line is not a sliver.
        val height = 36f
        val values = listOf(0.70000f, 0.70001f)  // diff ≈ 1e-5f in float32
        val rawRange = values.max() - values.min()
        assertTrue("rawRange must be above 1e-6f for this test to be meaningful",
            rawRange > 1e-6f)

        val range = safeRange(values)
        // Guard must NOT have activated — real range is used.
        assertEquals("near-flat: real range must be used (guard must not substitute 1f)",
            rawRange, range, 1e-9f)

        val ys = yPositions(values, height)
        val pad    = height * 0.1f
        val usable = height - 2f * pad

        // Both y-values must be finite and within the padded canvas bounds.
        assertTrue("all y-values must be finite", ys.all { it.isFinite() })
        assertTrue("all y-values must be >= pad",
            ys.all { it >= pad - 1e-4f })
        assertTrue("all y-values must be <= pad + usable",
            ys.all { it <= pad + usable + 1e-4f })

        // The visual span between the two points must equal the full usable height
        // because normalisation maps min→bottom and max→top.
        val visualSpan = kotlin.math.abs(ys[0] - ys[1])
        assertEquals(
            "near-flat: visual span must equal full usable height (normalisation gives full span)",
            usable, visualSpan, 1e-3f
        )
    }

    @Test
    fun `near-flat spread of 1e-4 maps min and max to the full usable height`() {
        val height = 36f
        val values = listOf(0.7000f, 0.7001f)  // diff = 1e-4f
        val range = safeRange(values)
        val rawRange = values.max() - values.min()
        assertTrue("rawRange must be above 1e-6f", rawRange > 1e-6f)
        assertEquals("real range must be used", rawRange, range, 1e-9f)

        val ys = yPositions(values, height)
        val usable = height - 2f * (height * 0.1f)

        assertTrue("y-values must be finite", ys.all { it.isFinite() })
        val visualSpan = kotlin.math.abs(ys[0] - ys[1])
        assertEquals(
            "near-flat 1e-4: visual span must equal full usable height",
            usable, visualSpan, 1e-3f
        )
    }

    @Test
    fun `near-flat multi-point trend stays bounded and has distinct y-positions at extremes`() {
        // Five points clustered very tightly — the min and max must still map to
        // the top and bottom of the usable area respectively.
        val height = 36f
        val values = listOf(0.70000f, 0.70002f, 0.70001f, 0.70003f, 0.70001f)
        val rawRange = values.max() - values.min()
        assertTrue("rawRange must be above 1e-6f", rawRange > 1e-6f)

        val ys = yPositions(values, height)
        val pad    = height * 0.1f
        val usable = height - 2f * pad

        assertTrue("all y-values must be finite", ys.all { it.isFinite() })
        assertTrue("all y-values must be within [pad, pad+usable]",
            ys.all { it >= pad - 1e-4f && it <= pad + usable + 1e-4f })

        // The extreme points (min/max values) must be at the edges of the usable area.
        val minIdx = values.indexOfFirst { it == values.min() }
        val maxIdx = values.indexOfFirst { it == values.max() }
        assertEquals("min-value point must be at bottom (pad + usable)",
            pad + usable, ys[minIdx], 1e-3f)
        assertEquals("max-value point must be at top (pad)",
            pad, ys[maxIdx], 1e-3f)
    }

    @Test
    fun `near-flat y-positions are all within the padded canvas area`() {
        // Confirm no y-value escapes the [pad, pad+usable] band for a near-flat trend.
        val height = 36f
        val values = listOf(0.700000f, 0.700005f, 0.700003f)
        val rawRange = values.max() - values.min()
        assertTrue("rawRange must be above 1e-6f", rawRange > 1e-6f)

        val ys = yPositions(values, height)
        val pad    = height * 0.1f
        val usable = height - 2f * pad

        ys.forEachIndexed { i, y ->
            assertTrue("y[$i]=$y must be >= pad ($pad)", y >= pad - 1e-4f)
            assertTrue("y[$i]=$y must be <= pad+usable (${pad + usable})", y <= pad + usable + 1e-4f)
        }
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
