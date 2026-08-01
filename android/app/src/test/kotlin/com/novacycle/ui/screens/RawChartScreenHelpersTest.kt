package com.novacycle.ui.screens

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Unit tests for [candleAgoText] (and, indirectly, the private
 * `parseIso8601ToMillis` it delegates to).
 *
 * `parseIso8601ToMillis` tries three formats in order:
 *   1. `yyyy-MM-dd'T'HH:mm:ssXXX`  — ISO datetime with timezone offset
 *   2. `yyyy-MM-dd'T'HH:mm:ss`     — bare ISO datetime (no offset)
 *   3. `yyyy-MM-dd`                 — date-only daily bar
 *
 * All "X ago" inputs are built relative to [System.currentTimeMillis] at
 * test time so there is no dependency on a fixed wall-clock date.
 *
 * For the bare-ISO format (no offset) the JVM default timezone governs
 * parsing; the test mirrors that by formatting via the same default timezone,
 * so the round-trip is always consistent.
 */
class RawChartScreenHelpersTest {

    // ── Format 1: ISO datetime with UTC offset ────────────────────────────────

    @Test
    fun `intraday UTC-offset timestamp two minutes ago produces 2m ago`() {
        val ts = isoWithOffset(System.currentTimeMillis() - 2 * 60_000L)
        assertEquals("2m ago", candleAgoText(ts))
    }

    @Test
    fun `intraday UTC-offset timestamp three hours ago produces 3h ago`() {
        val ts = isoWithOffset(System.currentTimeMillis() - 3 * 3_600_000L)
        assertEquals("3h ago", candleAgoText(ts))
    }

    @Test
    fun `intraday UTC-offset timestamp less than 60 seconds ago produces just now`() {
        val ts = isoWithOffset(System.currentTimeMillis() - 30_000L)
        assertEquals("just now", candleAgoText(ts))
    }

    // ── Format 2: bare ISO datetime (no timezone offset) ─────────────────────

    @Test
    fun `bare intraday timestamp without offset still produces a numeric result`() {
        // SimpleDateFormat without a timezone spec parses in the JVM default
        // timezone.  We format in that same default timezone so the round-trip
        // is consistent regardless of the host machine's locale.
        val fiveMinutesAgoMs = System.currentTimeMillis() - 5 * 60_000L
        val fmt = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
        // intentionally no timeZone override — mirrors parseIso8601ToMillis behaviour
        val ts = fmt.format(Date(fiveMinutesAgoMs))

        val result = candleAgoText(ts)
        assertNotNull("Expected a non-null result for bare ISO timestamp", result)
        assertTrue(
            "Expected a minutes/hours/just-now result, got: $result",
            result!!.matches(Regex("\\d+[mh] ago|just now"))
        )
    }

    // ── Format 3: date-only daily bar ─────────────────────────────────────────

    @Test
    fun `date-only daily bar from the past produces Xd ago`() {
        // "2024-01-01" is well in the past regardless of when this test runs.
        // parseIso8601ToMillis parses it as midnight in the JVM default timezone,
        // which is fine — it will always be more than one day behind now.
        val result = candleAgoText("2024-01-01")
        assertNotNull("Expected non-null for a past date-only timestamp", result)
        assertTrue(
            "Expected Xd ago, got: $result",
            result!!.matches(Regex("\\d+d ago"))
        )
    }

    // ── Future timestamp ──────────────────────────────────────────────────────

    @Test
    fun `future timestamp returns null`() {
        // A date far in the future; diffMs will be negative → null.
        val result = candleAgoText("2099-12-31T00:00:00+00:00")
        assertNull("Future timestamp should return null, got: $result", result)
    }

    // ── Null / unknown formats ────────────────────────────────────────────────

    @Test
    fun `unrecognised format string returns null gracefully`() {
        assertNull(candleAgoText("not-a-timestamp"))
    }

    @Test
    fun `empty string returns null gracefully`() {
        assertNull(candleAgoText(""))
    }

    @Test
    fun `partially valid string returns null gracefully`() {
        // Looks like a datetime but is subtly malformed (month 13).
        assertNull(candleAgoText("2026-13-01T10:00:00Z"))
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** Formats [epochMs] as `yyyy-MM-dd'T'HH:mm:ssXXX` in UTC. */
    private fun isoWithOffset(epochMs: Long): String =
        SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.US)
            .apply { timeZone = TimeZone.getTimeZone("UTC") }
            .format(Date(epochMs))
}
