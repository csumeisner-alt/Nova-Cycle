package com.novacycle.ui.components

import java.time.Instant
import org.junit.Assert.assertEquals
import org.junit.Test

class FormatTrainedAtTest {

    // Fixed "now": 2026-07-25T12:00:00Z
    private val nowMillis = Instant.parse("2026-07-25T12:00:00Z").toEpochMilli()

    @Test
    fun `Z-suffixed instant renders as relative age`() {
        val result = formatTrainedAt(nowMillis, "2026-07-25T09:00:00Z")
        assertEquals("3 h ago (2026-07-25T09:00:00Z)", result)
    }

    @Test
    fun `Z-suffixed instant with fractional seconds parses`() {
        // 11 min 59.876 s before now → floors to 11 min.
        val result = formatTrainedAt(nowMillis, "2026-07-25T11:48:00.123456Z")
        assertEquals("11 min ago (2026-07-25T11:48:00.123456Z)", result)
    }

    @Test
    fun `explicit positive offset is converted to UTC`() {
        // 11:00+02:00 == 09:00Z → 3 h before now
        val result = formatTrainedAt(nowMillis, "2026-07-25T11:00:00+02:00")
        assertEquals("3 h ago (2026-07-25T11:00:00+02:00)", result)
    }

    @Test
    fun `explicit negative offset is converted to UTC`() {
        // 05:00-05:00 == 10:00Z → 2 h before now
        val result = formatTrainedAt(nowMillis, "2026-07-25T05:00:00-05:00")
        assertEquals("2 h ago (2026-07-25T05:00:00-05:00)", result)
    }

    @Test
    fun `naive timestamp is treated as UTC and renders as relative age`() {
        // Python datetime.isoformat() with no timezone info.
        val result = formatTrainedAt(nowMillis, "2026-07-25T09:00:00")
        assertEquals("3 h ago (2026-07-25T09:00:00)", result)
    }

    @Test
    fun `naive timestamp with microseconds parses`() {
        val result = formatTrainedAt(nowMillis, "2026-07-25T11:59:30.654321")
        assertEquals("just now (2026-07-25T11:59:30.654321)", result)
    }

    @Test
    fun `naive timestamp from a previous day renders in days`() {
        val result = formatTrainedAt(nowMillis, "2026-07-23T12:00:00")
        assertEquals("2 days ago (2026-07-23T12:00:00)", result)
    }

    @Test
    fun `garbage input falls back to the raw string`() {
        assertEquals("not-a-date", formatTrainedAt(nowMillis, "not-a-date"))
    }

    @Test
    fun `empty string falls back to the raw string`() {
        assertEquals("", formatTrainedAt(nowMillis, ""))
    }

    @Test
    fun `date-only string is unparseable and falls back to raw`() {
        assertEquals("2026-07-25", formatTrainedAt(nowMillis, "2026-07-25"))
    }

    @Test
    fun `surrounding whitespace is tolerated for parsing`() {
        val result = formatTrainedAt(nowMillis, " 2026-07-25T09:00:00Z ")
        assertEquals("3 h ago ( 2026-07-25T09:00:00Z )", result)
    }

    @Test
    fun `future timestamp clamps to just now`() {
        val result = formatTrainedAt(nowMillis, "2026-07-25T13:00:00Z")
        assertEquals("just now (2026-07-25T13:00:00Z)", result)
    }
}
