package com.novacycle.ui.components

import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZonedDateTime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MarketAwareStalenessTest {

    private fun etMillis(dateTime: String): Long =
        ZonedDateTime.of(LocalDateTime.parse(dateTime), MarketHours.MARKET_ZONE)
            .toInstant().toEpochMilli()

    // ---- MarketHours ----

    @Test
    fun `open during a regular weekday session`() {
        // Wed Jul 22 2026, 11:00 ET
        assertTrue(MarketHours.isMarketOpen(etMillis("2026-07-22T11:00:00")))
        assertTrue(MarketHours.isMarketOpen(etMillis("2026-07-22T09:30:00")))
    }

    @Test
    fun `closed before open, at close, evenings and weekends`() {
        assertFalse(MarketHours.isMarketOpen(etMillis("2026-07-22T09:29:59")))
        assertFalse(MarketHours.isMarketOpen(etMillis("2026-07-22T16:00:00")))
        assertFalse(MarketHours.isMarketOpen(etMillis("2026-07-22T20:00:00")))
        // Sat/Sun Jul 25-26 2026
        assertFalse(MarketHours.isMarketOpen(etMillis("2026-07-25T11:00:00")))
        assertFalse(MarketHours.isMarketOpen(etMillis("2026-07-26T11:00:00")))
    }

    @Test
    fun `closed on market holidays`() {
        // Jul 3 2026 is a Friday: Independence Day (Sat Jul 4) observed
        assertFalse(MarketHours.isMarketOpen(etMillis("2026-07-03T11:00:00")))
        assertTrue(MarketHours.isMarketHoliday(LocalDate.of(2026, 1, 1)))   // New Year's
        assertTrue(MarketHours.isMarketHoliday(LocalDate.of(2026, 1, 19)))  // MLK (3rd Mon)
        assertTrue(MarketHours.isMarketHoliday(LocalDate.of(2026, 4, 3)))   // Good Friday
        assertTrue(MarketHours.isMarketHoliday(LocalDate.of(2026, 5, 25)))  // Memorial Day
        assertTrue(MarketHours.isMarketHoliday(LocalDate.of(2026, 11, 26))) // Thanksgiving
        assertTrue(MarketHours.isMarketHoliday(LocalDate.of(2026, 12, 25))) // Christmas
        assertFalse(MarketHours.isMarketHoliday(LocalDate.of(2026, 7, 22)))
    }

    @Test
    fun `previous session close skips weekends and holidays`() {
        val fridayClose = etMillis("2026-07-24T16:00:00")
        // Saturday morning -> Friday's close
        assertEquals(fridayClose, MarketHours.previousSessionCloseMillis(etMillis("2026-07-25T09:00:00")))
        // Sunday night -> still Friday's close
        assertEquals(fridayClose, MarketHours.previousSessionCloseMillis(etMillis("2026-07-26T23:00:00")))
        // Mid-session Monday -> Friday's close (today hasn't closed yet)
        assertEquals(fridayClose, MarketHours.previousSessionCloseMillis(etMillis("2026-07-27T11:00:00")))
        // Monday evening -> Monday's close
        assertEquals(etMillis("2026-07-27T16:00:00"),
            MarketHours.previousSessionCloseMillis(etMillis("2026-07-27T18:00:00")))
        // Sat Jul 4 weekend 2026: Fri Jul 3 is the observed holiday -> Thu Jul 2 close
        assertEquals(etMillis("2026-07-02T16:00:00"),
            MarketHours.previousSessionCloseMillis(etMillis("2026-07-05T12:00:00")))
    }

    // ---- marketAwareStalenessLevel ----

    @Test
    fun `during market hours behavior is unchanged`() {
        val now = etMillis("2026-07-22T11:00:00")
        assertEquals(StalenessLevel.FRESH, marketAwareStalenessLevel(now, now - 4 * 60_000L))
        assertEquals(StalenessLevel.WARNING, marketAwareStalenessLevel(now, now - 5 * 60_000L))
        assertEquals(StalenessLevel.CRITICAL, marketAwareStalenessLevel(now, now - 15 * 60_000L))
    }

    @Test
    fun `data refreshed near the close stays fresh all evening and weekend`() {
        val closeUpdate = etMillis("2026-07-24T15:58:00") // Fri, 2 min before close
        assertEquals(StalenessLevel.FRESH,
            marketAwareStalenessLevel(etMillis("2026-07-24T22:00:00"), closeUpdate)) // Fri night
        assertEquals(StalenessLevel.FRESH,
            marketAwareStalenessLevel(etMillis("2026-07-26T11:00:00"), closeUpdate)) // Sunday
    }

    @Test
    fun `data already stale at the close still warns off-hours`() {
        val staleUpdate = etMillis("2026-07-24T15:30:00") // 30 min before Fri close
        assertEquals(StalenessLevel.CRITICAL,
            marketAwareStalenessLevel(etMillis("2026-07-25T12:00:00"), staleUpdate)) // Saturday
        val slightlyStale = etMillis("2026-07-24T15:52:00") // 8 min before close
        assertEquals(StalenessLevel.WARNING,
            marketAwareStalenessLevel(etMillis("2026-07-25T12:00:00"), slightlyStale))
    }

    @Test
    fun `stale data goes red again once the market reopens`() {
        val closeUpdate = etMillis("2026-07-24T15:59:00")
        // Monday 10:00 ET, no refresh since Friday -> critical during open hours
        assertEquals(StalenessLevel.CRITICAL,
            marketAwareStalenessLevel(etMillis("2026-07-27T10:00:00"), closeUpdate))
    }

    @Test
    fun `holiday counts as off-hours`() {
        val closeUpdate = etMillis("2026-07-02T15:59:00") // Thu before July 4th weekend
        // Fri Jul 3 (observed holiday) midday -> fresh
        assertEquals(StalenessLevel.FRESH,
            marketAwareStalenessLevel(etMillis("2026-07-03T12:00:00"), closeUpdate))
    }
}
