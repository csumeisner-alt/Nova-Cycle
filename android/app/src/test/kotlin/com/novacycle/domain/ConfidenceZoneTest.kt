package com.novacycle.domain

import com.novacycle.data.remote.models.PredictionResponse
import com.novacycle.domain.model.ConfidenceZone
import com.novacycle.domain.model.GaugeState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the normalized-confidence zone mapping and the neutral
 * fallback contract (Task: 0–100% gauge confidence with BUY/SELL/HOLD labels).
 */
class ConfidenceZoneTest {

    // ── Zone boundaries: 0–30 red/Weak, 31–64 yellow/Uncertain, 65–100 green/Strong ──

    @Test
    fun `zone boundaries map correctly`() {
        assertEquals(ConfidenceZone.WEAK, ConfidenceZone.fromPercent(0))
        assertEquals(ConfidenceZone.WEAK, ConfidenceZone.fromPercent(30))
        assertEquals(ConfidenceZone.UNCERTAIN, ConfidenceZone.fromPercent(31))
        assertEquals(ConfidenceZone.UNCERTAIN, ConfidenceZone.fromPercent(64))
        assertEquals(ConfidenceZone.STRONG, ConfidenceZone.fromPercent(65))
        assertEquals(ConfidenceZone.STRONG, ConfidenceZone.fromPercent(100))
    }

    @Test
    fun `out of range percents are clamped`() {
        assertEquals(ConfidenceZone.WEAK, ConfidenceZone.fromPercent(-50))
        assertEquals(ConfidenceZone.STRONG, ConfidenceZone.fromPercent(150))
    }

    @Test
    fun `zone labels are user friendly`() {
        assertEquals("Weak", ConfidenceZone.WEAK.label)
        assertEquals("Uncertain", ConfidenceZone.UNCERTAIN.label)
        assertEquals("Strong", ConfidenceZone.STRONG.label)
    }

    // ── GaugeState defaults & fallback contract ──────────────────────────

    @Test
    fun `default gauge state is neutral hold`() {
        val state = GaugeState()
        assertEquals(0, state.confidencePercent)
        assertEquals("NEUTRAL", state.trend)
        assertEquals("NEUTRAL / HOLD", state.displaySignal)
        assertEquals(ConfidenceZone.WEAK, state.confidenceZone)
    }

    @Test
    fun `gauge state zone follows confidence percent`() {
        assertEquals(ConfidenceZone.STRONG, GaugeState(confidencePercent = 72).confidenceZone)
        assertEquals(ConfidenceZone.UNCERTAIN, GaugeState(confidencePercent = 50).confidenceZone)
    }

    @Test
    fun `directional gauge maps raw score to left to right percentage`() {
        assertEquals(0, GaugeState(score = -100f).gaugePercent)
        assertEquals(38, GaugeState(score = -24f).gaugePercent)
        assertEquals(50, GaugeState(score = 0f).gaugePercent)
        assertEquals(60, GaugeState(score = 20f).gaugePercent)
        assertEquals(100, GaugeState(score = 100f).gaugePercent)
    }

    @Test
    fun `directional action uses the same gauge percentage`() {
        assertEquals("SELL", GaugeState(score = -100f).gaugeAction)
        assertEquals("HOLD", GaugeState(score = -24f).gaugeAction)
        assertEquals("HOLD", GaugeState(score = 0f).gaugeAction)
        assertEquals("BUY", GaugeState(score = 100f).gaugeAction)
    }

    // ── PredictionResponse: missing fields default to neutral (old backend) ──

    @Test
    fun `prediction response defaults to neutral when new fields absent`() {
        val old = PredictionResponse(score = 42f, signal = "buy", confidence = 0.42f)
        assertEquals(0, old.confidencePercent)
        assertEquals("NEUTRAL", old.trend)
        assertEquals("NEUTRAL / HOLD", old.displaySignal)
    }

    @Test
    fun `fallback prediction is neutral hold`() {
        val fallback = com.novacycle.viewmodel.DualGaugeViewModel.NEUTRAL_FALLBACK
        assertEquals(0, fallback.confidencePercent)
        assertEquals("NEUTRAL", fallback.trend)
        assertEquals("NEUTRAL / HOLD", fallback.displaySignal)
        assertTrue(fallback.note != null)
    }
}
