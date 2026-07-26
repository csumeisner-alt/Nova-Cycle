package com.novacycle.ui.components

import java.time.DayOfWeek
import java.time.Instant
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * US equity market (NYSE/Nasdaq) regular-session calendar used to decide
 * whether data is *expected* to be updating right now.
 *
 * Regular session: Mon-Fri 09:30-16:00 America/New_York, excluding full-day
 * market holidays. Early-close half days are treated as normal days (the
 * label just goes quiet a bit earlier than strictly necessary, which is the
 * safe direction).
 */
object MarketHours {

    val MARKET_ZONE: ZoneId = ZoneId.of("America/New_York")
    val SESSION_OPEN: LocalTime = LocalTime.of(9, 30)
    val SESSION_CLOSE: LocalTime = LocalTime.of(16, 0)

    /** True when the regular US equity session is open at [epochMillis]. */
    fun isMarketOpen(epochMillis: Long, zone: ZoneId = MARKET_ZONE): Boolean {
        val zdt = Instant.ofEpochMilli(epochMillis).atZone(zone)
        if (!isTradingDay(zdt.toLocalDate())) return false
        val time = zdt.toLocalTime()
        return !time.isBefore(SESSION_OPEN) && time.isBefore(SESSION_CLOSE)
    }

    /**
     * Epoch millis of the most recent regular-session close at or before
     * [epochMillis]. If the session is currently open, this is the close of
     * the *previous* trading day.
     */
    fun previousSessionCloseMillis(epochMillis: Long, zone: ZoneId = MARKET_ZONE): Long {
        val zdt = Instant.ofEpochMilli(epochMillis).atZone(zone)
        var date = zdt.toLocalDate()
        // Today's close counts only if today trades and we're at/past close.
        if (!(isTradingDay(date) && !zdt.toLocalTime().isBefore(SESSION_CLOSE))) {
            date = date.minusDays(1)
            while (!isTradingDay(date)) date = date.minusDays(1)
        }
        return ZonedDateTime.of(date, SESSION_CLOSE, zone).toInstant().toEpochMilli()
    }

    /** Weekday and not a full-day market holiday. */
    fun isTradingDay(date: LocalDate): Boolean =
        date.dayOfWeek != DayOfWeek.SATURDAY &&
            date.dayOfWeek != DayOfWeek.SUNDAY &&
            !isMarketHoliday(date)

    /**
     * Full-day NYSE holidays: New Year's Day, MLK Day, Presidents' Day,
     * Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day,
     * Thanksgiving, Christmas — with weekend observance shifts for the
     * fixed-date ones.
     */
    fun isMarketHoliday(date: LocalDate): Boolean {
        val y = date.year
        val fixed = listOf(
            LocalDate.of(y, 1, 1),   // New Year's Day
            LocalDate.of(y, 6, 19),  // Juneteenth
            LocalDate.of(y, 7, 4),   // Independence Day
            LocalDate.of(y, 12, 25)  // Christmas
        )
        if (fixed.any { observed(it) == date }) return true
        // New Year's observed on Dec 31 when Jan 1 (next year) is a Saturday.
        if (observed(LocalDate.of(y + 1, 1, 1)) == date) return true

        return date == nthWeekdayOfMonth(y, 1, DayOfWeek.MONDAY, 3) ||   // MLK Day
            date == nthWeekdayOfMonth(y, 2, DayOfWeek.MONDAY, 3) ||      // Presidents' Day
            date == goodFriday(y) ||
            date == lastWeekdayOfMonth(y, 5, DayOfWeek.MONDAY) ||        // Memorial Day
            date == nthWeekdayOfMonth(y, 9, DayOfWeek.MONDAY, 1) ||      // Labor Day
            date == nthWeekdayOfMonth(y, 11, DayOfWeek.THURSDAY, 4)      // Thanksgiving
    }

    /** Saturday holidays observed Friday; Sunday holidays observed Monday. */
    private fun observed(date: LocalDate): LocalDate = when (date.dayOfWeek) {
        DayOfWeek.SATURDAY -> date.minusDays(1)
        DayOfWeek.SUNDAY -> date.plusDays(1)
        else -> date
    }

    private fun nthWeekdayOfMonth(year: Int, month: Int, dow: DayOfWeek, n: Int): LocalDate {
        var d = LocalDate.of(year, month, 1)
        while (d.dayOfWeek != dow) d = d.plusDays(1)
        return d.plusWeeks((n - 1).toLong())
    }

    private fun lastWeekdayOfMonth(year: Int, month: Int, dow: DayOfWeek): LocalDate {
        var d = LocalDate.of(year, month, 1).plusMonths(1).minusDays(1)
        while (d.dayOfWeek != dow) d = d.minusDays(1)
        return d
    }

    /** Good Friday = Easter Sunday (Gregorian computus) minus 2 days. */
    private fun goodFriday(year: Int): LocalDate {
        val a = year % 19
        val b = year / 100
        val c = year % 100
        val d = b / 4
        val e = b % 4
        val f = (b + 8) / 25
        val g = (b - f + 1) / 3
        val h = (19 * a + b - d - g + 15) % 30
        val i = c / 4
        val k = c % 4
        val l = (32 + 2 * e + 2 * i - h - k) % 7
        val m = (a + 11 * h + 22 * l) / 451
        val month = (h + l - 7 * m + 114) / 31
        val day = ((h + l - 7 * m + 114) % 31) + 1
        return LocalDate.of(year, month, day).minusDays(2)
    }
}
