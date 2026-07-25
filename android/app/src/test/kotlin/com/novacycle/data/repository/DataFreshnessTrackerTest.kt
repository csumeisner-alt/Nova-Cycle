package com.novacycle.data.repository

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Verifies the data-freshness contract behind the "last updated X ago" label:
 * only explicit data-fetch recordings advance the timestamp. Health polls
 * never call [DataFreshnessTracker.recordSuccess] (see HealthViewModel /
 * NovaCycleRepository.getHealth), so a reachable-but-idle backend cannot make
 * on-screen data look fresher than it is.
 */
class DataFreshnessTrackerTest {

    @Test
    fun `starts with no freshness timestamp`() {
        val tracker = DataFreshnessTracker()
        assertNull(tracker.lastSuccessAtMillis.value)
    }

    @Test
    fun `recordSuccess sets the timestamp`() {
        val tracker = DataFreshnessTracker()
        tracker.recordSuccess(nowMillis = 1_000L)
        assertEquals(1_000L, tracker.lastSuccessAtMillis.value)
    }

    @Test
    fun `timestamp only advances when a data fetch is recorded`() {
        val tracker = DataFreshnessTracker()
        tracker.recordSuccess(nowMillis = 1_000L)
        // Simulated time passing with successful health polls but no data
        // fetches: nothing calls recordSuccess, so the timestamp is unchanged
        // and the UI reports the true age of on-screen data.
        assertEquals(1_000L, tracker.lastSuccessAtMillis.value)
        tracker.recordSuccess(nowMillis = 5_000L)
        assertEquals(5_000L, tracker.lastSuccessAtMillis.value)
    }
}

/** Deterministic checks for the relative-age formatting shown in the banner. */
class RelativeAgeFormatTest {

    private fun fmt(ageMillis: Long): String =
        com.novacycle.ui.components.formatRelativeAge(nowMillis = ageMillis, thenMillis = 0L)

    @Test
    fun `formats ages across ranges`() {
        assertEquals("just now", fmt(30_000L))
        assertEquals("12 min ago", fmt(12 * 60_000L))
        assertEquals("3 h 5 min ago", fmt((3 * 60 + 5) * 60_000L))
        assertEquals("2 h ago", fmt(2 * 60 * 60_000L))
        assertEquals("2 days ago", fmt(2 * 24 * 60 * 60_000L))
    }

    @Test
    fun `clock skew never shows negative age`() {
        assertEquals("just now", fmt(-60_000L))
    }
}
