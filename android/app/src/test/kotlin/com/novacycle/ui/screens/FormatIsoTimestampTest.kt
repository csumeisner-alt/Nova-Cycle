package com.novacycle.ui.screens

import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.ZoneId

/**
 * Unit tests for [formatIsoTimestamp].
 *
 * Every test pins [ZoneId] to "America/New_York" so results are deterministic
 * regardless of the host machine's system timezone. This also exercises the
 * DST-awareness requirement: if someone accidentally applies a fixed UTC-5 offset
 * instead of letting the zone rule govern the conversion, the post-DST test will
 * catch it (the answer would be off by one hour).
 *
 * ## DST reference points used here
 * In 2026, America/New_York springs forward at 02:00 EST on Sunday 8 March
 * (clocks jump to 03:00 EDT = UTC-4). The first parameterized pair therefore
 * uses timestamps that straddle that boundary:
 *
 *   "2026-03-08T06:30:00Z" → 01:30 AM EST  (UTC-5, before the switch)
 *   "2026-03-08T08:30:00Z" → 04:30 AM EDT  (UTC-4, after the switch)
 *
 * A naïve (no-Z) string is passed through the ISO_DATE_TIME formatter as-is
 * local time — no zone conversion occurs — so it is formatted directly.
 */
class FormatIsoTimestampTest {

    private val newYork = ZoneId.of("America/New_York")

    // ── UTC "Z" strings ───────────────────────────────────────────────────────

    @Test
    fun `UTC timestamp in EST (before DST) converts correctly`() {
        // 06:30 UTC on 2026-03-08 = 01:30 AM EST (UTC-5). DST has not yet
        // kicked in — clocks spring forward at 07:00 UTC (= 02:00 AM local).
        val result = formatIsoTimestamp("2026-03-08T06:30:00Z", zone = newYork)
        assertEquals("2026-03-08 01:30", result)
    }

    @Test
    fun `UTC timestamp in EDT (after DST spring-forward) converts correctly`() {
        // 08:30 UTC on 2026-03-08 = 04:30 AM EDT (UTC-4). DST is now active.
        // A bug that always uses EST (UTC-5) would produce "03:30" — off by one hour.
        val result = formatIsoTimestamp("2026-03-08T08:30:00Z", zone = newYork)
        assertEquals("2026-03-08 04:30", result)
    }

    @Test
    fun `regular UTC summer timestamp converts to EDT`() {
        // A mid-summer date: 2026-07-01T14:00:00Z = 10:00 AM EDT (UTC-4).
        val result = formatIsoTimestamp("2026-07-01T14:00:00Z", zone = newYork)
        assertEquals("2026-07-01 10:00", result)
    }

    @Test
    fun `regular UTC winter timestamp converts to EST`() {
        // A mid-winter date: 2026-01-15T18:00:00Z = 01:00 PM EST (UTC-5).
        val result = formatIsoTimestamp("2026-01-15T18:00:00Z", zone = newYork)
        assertEquals("2026-01-15 13:00", result)
    }

    // ── Naïve ISO strings (no "Z" suffix) ────────────────────────────────────

    @Test
    fun `naive ISO string without Z suffix is formatted as local time without zone shift`() {
        // "2026-07-01T10:30:00.123456" has no UTC indicator → treated as local
        // time and formatted directly. No zone conversion should occur.
        val result = formatIsoTimestamp("2026-07-01T10:30:00.123456", zone = newYork)
        assertEquals("2026-07-01 10:30", result)
    }

    @Test
    fun `naive ISO string with no fractional seconds formats correctly`() {
        val result = formatIsoTimestamp("2026-03-08T15:45:00", zone = newYork)
        assertEquals("2026-03-08 15:45", result)
    }

    // ── Edge / fallback cases ─────────────────────────────────────────────────

    @Test
    fun `null input returns double-dash placeholder`() {
        val result = formatIsoTimestamp(null, zone = newYork)
        assertEquals("--", result)
    }

    @Test
    fun `unparseable string is returned unchanged`() {
        val garbled = "not-a-date"
        val result = formatIsoTimestamp(garbled, zone = newYork)
        assertEquals(garbled, result)
    }

    @Test
    fun `empty string is returned unchanged`() {
        val result = formatIsoTimestamp("", zone = newYork)
        assertEquals("", result)
    }
}
