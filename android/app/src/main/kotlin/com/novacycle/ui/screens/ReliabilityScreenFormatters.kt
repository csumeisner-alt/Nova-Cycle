package com.novacycle.ui.screens

import java.time.Instant
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

internal val displayFormatter: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")

/**
 * Converts a UTC ISO timestamp string (e.g. "2026-07-14T01:06:47Z") or a naive
 * local ISO string (e.g. "2026-07-14T01:06:47.323043") to a human-readable
 * local date-time string using [zone].
 *
 * The [zone] parameter defaults to [ZoneId.systemDefault] so callers in the UI
 * pass nothing, but unit tests may supply a pinned zone to verify DST behaviour.
 *
 * Falls back to returning [iso] unchanged rather than crashing on unparseable input.
 */
internal fun formatIsoTimestamp(
    iso: String?,
    zone: ZoneId = ZoneId.systemDefault()
): String {
    if (iso == null) return "--"
    return try {
        // Try offset/zoned ISO first (e.g. 2026-07-14T01:06:47Z)
        val instant = Instant.parse(iso)
        LocalDateTime.ofInstant(instant, zone).format(displayFormatter)
    } catch (e: Exception) {
        try {
            // Fall back to naive local ISO (e.g. 2026-07-14T01:06:47.323043)
            LocalDateTime.parse(iso, DateTimeFormatter.ISO_DATE_TIME).format(displayFormatter)
        } catch (e2: Exception) {
            iso
        }
    }
}
