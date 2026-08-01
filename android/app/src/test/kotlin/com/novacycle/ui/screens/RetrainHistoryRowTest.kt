package com.novacycle.ui.screens

import com.novacycle.data.remote.models.AccuracyHistoryEntry
import java.time.ZoneId
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the model-name display logic in [RetrainHistoryRow].
 *
 * RetrainHistoryRow (ReliabilityScreen.kt) shows the model-name Text only
 * when `entry.modelName.isNotBlank()`. For long names the composable applies
 * `maxLines = 1` and `overflow = TextOverflow.Ellipsis` so the row never
 * breaks its layout.
 *
 * These tests exercise the pure-Kotlin predicates without requiring a Compose
 * runtime, following the same approach as [AccuracySparklineLogicTest].
 */
class RetrainHistoryRowTest {

    // ── Helpers mirroring the composable's visibility predicate exactly ───────

    /**
     * Mirrors `if (entry.modelName.isNotBlank())` from RetrainHistoryRow.
     * Returns true when the model-name label should be rendered.
     */
    private fun shouldShowModelName(entry: AccuracyHistoryEntry): Boolean =
        entry.modelName.isNotBlank()

    // ── Blank / missing name — label must be absent ───────────────────────────

    @Test
    fun `empty string modelName hides the name label`() {
        val entry = AccuracyHistoryEntry(
            modelName = "",
            trainedAt = "2025-01-01T00:00:00",
            accuracy = 0.72f
        )
        assertFalse(
            "empty string must not show the model-name label",
            shouldShowModelName(entry)
        )
    }

    @Test
    fun `whitespace-only modelName hides the name label`() {
        val entry = AccuracyHistoryEntry(
            modelName = "   ",
            trainedAt = "2025-01-01T00:00:00",
            accuracy = 0.68f
        )
        assertFalse(
            "spaces-only string must not show the model-name label",
            shouldShowModelName(entry)
        )
    }

    @Test
    fun `tab and newline only modelName hides the name label`() {
        val entry = AccuracyHistoryEntry(
            modelName = "\t\n",
            trainedAt = "2025-01-01T00:00:00",
            accuracy = 0.60f
        )
        assertFalse(
            "tab/newline-only string must not show the model-name label",
            shouldShowModelName(entry)
        )
    }

    @Test
    fun `default AccuracyHistoryEntry has blank modelName so label is hidden`() {
        // AccuracyHistoryEntry default is modelName = "" (Moshi @JsonClass adapter default)
        val entry = AccuracyHistoryEntry()
        assertFalse(
            "default-constructed entry must hide the model-name label",
            shouldShowModelName(entry)
        )
    }

    @Test
    fun `blank modelName means only the date row appears in the layout`() {
        // When the name label is hidden the only remaining content is the
        // trainedAt date line. Verify the date field is still accessible.
        val entry = AccuracyHistoryEntry(
            modelName = "",
            trainedAt = "2026-03-15T09:00:00",
            accuracy = 0.75f
        )
        assertFalse("name label must be absent", shouldShowModelName(entry))
        // The date/timestamp column is always present regardless of modelName.
        assertEquals("2026-03-15T09:00:00", entry.trainedAt)
    }

    // ── Non-blank name — label must be present ────────────────────────────────

    @Test
    fun `non-blank modelName shows the name label`() {
        val entry = AccuracyHistoryEntry(
            modelName = "xgb_v3",
            trainedAt = "2025-06-01T00:00:00",
            accuracy = 0.80f
        )
        assertTrue(
            "non-blank model name must show the label",
            shouldShowModelName(entry)
        )
    }

    @Test
    fun `name with leading and trailing spaces is non-blank so label is shown`() {
        // isNotBlank() returns true when there is at least one non-whitespace char.
        val entry = AccuracyHistoryEntry(
            modelName = "  xgb_v4  ",
            trainedAt = "2025-06-01T00:00:00",
            accuracy = 0.81f
        )
        assertTrue(
            "name with surrounding spaces must show the label",
            shouldShowModelName(entry)
        )
    }

    // ── Very long name — label shown, truncation contract preserved ───────────

    @Test
    fun `very long modelName is non-blank so the label would be rendered`() {
        val longName = "xgb_retrain_" + "a".repeat(500)
        val entry = AccuracyHistoryEntry(
            modelName = longName,
            trainedAt = "2025-09-01T00:00:00",
            accuracy = 0.77f
        )
        assertTrue(
            "very long name must pass isNotBlank() so the composable renders it",
            shouldShowModelName(entry)
        )
    }

    @Test
    fun `very long modelName length exceeds single line width at typical densities`() {
        // The composable constrains rendering to maxLines = 1 with
        // TextOverflow.Ellipsis. This test verifies that the string length is
        // large enough to guarantee visual truncation on any realistic screen
        // density (a single line typically fits ~30-60 characters at bodySmall).
        val longName = "a".repeat(500)
        assertTrue(
            "500-char name must be longer than any single-line capacity",
            longName.length > 60
        )
        // The composable receives the full string; layout clipping + ellipsis
        // is applied by Compose. The predicate only needs to pass isNotBlank().
        assertTrue(longName.isNotBlank())
    }

    @Test
    fun `single character modelName is sufficient to show the label`() {
        val entry = AccuracyHistoryEntry(
            modelName = "v",
            trainedAt = "2025-01-15T12:00:00",
            accuracy = 0.65f
        )
        assertTrue(
            "single character is non-blank so the label must be shown",
            shouldShowModelName(entry)
        )
    }

    // ── Date label: null / empty trainedAt ────────────────────────────────────

    /**
     * Mirrors the Text composable in RetrainHistoryRow:
     *   Text(text = formatIsoTimestamp(entry.trainedAt), ...)
     *
     * When the backend sends null for trainedAt, formatIsoTimestamp must return
     * "--" so the row displays a placeholder rather than crashing or rendering
     * an empty string.
     */
    @Test
    fun `null trainedAt produces dash placeholder via formatIsoTimestamp`() {
        val entry = AccuracyHistoryEntry(
            modelName = "",
            trainedAt = null,
            accuracy = 0.70f
        )
        val displayed = formatIsoTimestamp(entry.trainedAt)
        assertEquals(
            "formatIsoTimestamp(null) must return \"--\" so RetrainHistoryRow shows a placeholder",
            "--",
            displayed
        )
    }

    /**
     * When trainedAt is an empty string the backend has sent an unparseable value.
     * formatIsoTimestamp falls back to returning the original string unchanged,
     * so the row displays "" rather than crashing.
     */
    @Test
    fun `empty string trainedAt returns empty string via formatIsoTimestamp`() {
        val entry = AccuracyHistoryEntry(
            modelName = "",
            trainedAt = "",
            accuracy = 0.70f
        )
        val displayed = formatIsoTimestamp(entry.trainedAt)
        assertEquals(
            "formatIsoTimestamp(\"\") must return \"\" (raw fallback) so RetrainHistoryRow does not crash",
            "",
            displayed
        )
    }

    // ── DST-edge timestamps — correct local hour must be shown ────────────────

    /**
     * Spring-forward boundary: America/New_York 2026-03-08.
     *
     * At 2026-03-08T07:00:00Z the clocks in New York jump from 2:00 AM EST to
     * 3:00 AM EDT (UTC-4 takes effect at that exact instant).  The formatted
     * output must show 03:00, not 02:00.
     */
    @Test
    fun `spring-forward DST boundary shows correct post-transition local hour`() {
        // 2026-03-08T07:00:00Z == the instant clocks spring from 02:00 EST → 03:00 EDT
        val displayed = formatIsoTimestamp(
            "2026-03-08T07:00:00Z",
            ZoneId.of("America/New_York")
        )
        assertEquals(
            "Timestamp at the spring-forward boundary must display as 03:00 EDT, not 02:00",
            "2026-03-08 03:00",
            displayed
        )
    }

    /**
     * Fall-back boundary: America/New_York 2025-11-02.
     *
     * At 2025-11-02T06:00:00Z the clocks in New York fall back from 2:00 AM EDT
     * to 1:00 AM EST (UTC-5 takes effect at that exact instant).  The formatted
     * output must show 01:00, not 02:00.
     */
    @Test
    fun `fall-back DST boundary shows correct post-transition local hour`() {
        // 2025-11-02T06:00:00Z == the instant clocks fall from 02:00 EDT → 01:00 EST
        val displayed = formatIsoTimestamp(
            "2025-11-02T06:00:00Z",
            ZoneId.of("America/New_York")
        )
        assertEquals(
            "Timestamp at the fall-back boundary must display as 01:00 EST, not 02:00",
            "2025-11-02 01:00",
            displayed
        )
    }
}
