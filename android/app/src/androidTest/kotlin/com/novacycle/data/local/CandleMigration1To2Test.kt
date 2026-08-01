package com.novacycle.data.local

import android.database.sqlite.SQLiteDatabase
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.novacycle.di.AppModule
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented migration test: v1 → v2 candle schema upgrade.
 *
 * The MIGRATION_1_2 in AppModule rebuilds the candles table, adding a
 * `timeframe` column and changing the primary key from (ticker, timestamp)
 * to (ticker, timeframe, timestamp). All pre-existing rows were daily bars
 * and must survive the upgrade as timeframe='daily'.
 *
 * Run locally:
 *   ./gradlew :app:connectedDebugAndroidTest \
 *     --tests "com.novacycle.data.local.CandleMigration1To2Test"
 *
 * Requires a connected device or running emulator. The test does NOT use
 * MigrationTestHelper.createDatabase() (which needs schema JSON assets) —
 * it constructs the v1 schema directly with raw SQL so it runs immediately
 * without needing a prior kapt pass to generate the JSON files.
 *
 * To generate the schema JSON files for future MigrationTestHelper use:
 *   ./gradlew :app:kaptDebugKotlin
 * then commit the files under android/app/schemas/.
 */
@RunWith(AndroidJUnit4::class)
class CandleMigration1To2Test {

    private val dbName = "candle_migration_1_2_test.db"
    private val context = ApplicationProvider.getApplicationContext<android.content.Context>()

    @Before
    fun setUp() {
        context.deleteDatabase(dbName)
    }

    @After
    fun tearDown() {
        context.deleteDatabase(dbName)
    }

    /**
     * Creates a v1-shaped database using raw SQLite, inserts two candle rows,
     * then opens it with Room (applying MIGRATION_1_2 and MIGRATION_2_3) and
     * verifies both rows are accessible through the DAO with timeframe='daily'.
     */
    @Test
    fun migration1To2_preservesExistingCandlesAsDailyTimeframe() {
        // ── Step 1: build a hand-crafted v1 database ──────────────────────────
        val dbFile = context.getDatabasePath(dbName)
        dbFile.parentFile?.mkdirs()

        val v1 = SQLiteDatabase.openOrCreateDatabase(dbFile, null)
        v1.execSQL("PRAGMA user_version = 1")

        // v1 candles: no `timeframe` column; PK is (ticker, timestamp)
        v1.execSQL(
            """
            CREATE TABLE IF NOT EXISTS candles (
                ticker               TEXT    NOT NULL,
                timestamp            TEXT    NOT NULL,
                open                 REAL    NOT NULL,
                high                 REAL    NOT NULL,
                low                  REAL    NOT NULL,
                close                REAL    NOT NULL,
                volume               INTEGER NOT NULL,
                is_extended_hours    INTEGER NOT NULL,
                session_type         TEXT    NOT NULL,
                gap_percent          REAL,
                gap_type             TEXT,
                PRIMARY KEY(ticker, timestamp)
            )
            """.trimIndent()
        )

        // v1 signal_history: no conviction columns (added by MIGRATION_2_3)
        v1.execSQL(
            """
            CREATE TABLE IF NOT EXISTS signal_history (
                id                       TEXT    NOT NULL,
                timestamp                TEXT    NOT NULL,
                ticker                   TEXT    NOT NULL,
                cycle_id                 TEXT,
                signal_type              TEXT    NOT NULL,
                gauge_type               TEXT    NOT NULL,
                confidence               REAL    NOT NULL,
                session_type             TEXT    NOT NULL,
                is_extended_hours        INTEGER NOT NULL,
                gap_type                 TEXT,
                liquidity_score          REAL    NOT NULL,
                macro_override_applied   INTEGER NOT NULL,
                PRIMARY KEY(id)
            )
            """.trimIndent()
        )

        // confidence_history is unchanged across all versions
        v1.execSQL(
            """
            CREATE TABLE IF NOT EXISTS confidence_history (
                id                       TEXT    NOT NULL,
                timestamp                TEXT    NOT NULL,
                ticker                   TEXT    NOT NULL,
                long_buy_confidence      REAL    NOT NULL,
                long_sell_confidence     REAL    NOT NULL,
                short_buy_confidence     REAL    NOT NULL,
                short_sell_confidence    REAL    NOT NULL,
                session_type             TEXT    NOT NULL,
                is_extended_hours        INTEGER NOT NULL,
                PRIMARY KEY(id)
            )
            """.trimIndent()
        )

        // Insert two daily candle rows — these must survive the migration
        v1.execSQL(
            """
            INSERT INTO candles VALUES
                ('VOO','2024-01-02',450.0,455.0,448.0,452.5,1000000,0,'regular',NULL,NULL)
            """.trimIndent()
        )
        v1.execSQL(
            """
            INSERT INTO candles VALUES
                ('VOO','2024-01-03',452.5,458.0,451.0,456.0,1200000,0,'regular',0.55,'gap_up')
            """.trimIndent()
        )
        v1.close()

        // ── Step 2: open with Room, running both migrations ────────────────────
        // Room sees user_version=1, applies MIGRATION_1_2 then MIGRATION_2_3,
        // then validates the resulting schema matches the v3 entity definitions.
        // If the migration SQL ever drifts from the entity definitions Room will
        // throw an IllegalStateException here — that's the guard this test provides.
        val db = Room.databaseBuilder(
            context,
            NovaCycleDatabase::class.java,
            dbName
        )
            .addMigrations(AppModule.MIGRATION_1_2, AppModule.MIGRATION_2_3)
            // fallbackToDestructiveMigration is intentionally absent here: if the
            // migration is missing or broken the test must fail, not silently wipe data.
            .build()

        try {
            // ── Step 3: verify candle rows survived with timeframe='daily' ─────
            val cursor = db.openHelper.readableDatabase.query(
                "SELECT ticker, timeframe, timestamp, close FROM candles ORDER BY timestamp"
            )

            val rows = mutableListOf<Map<String, String>>()
            while (cursor.moveToNext()) {
                rows.add(
                    mapOf(
                        "ticker"    to cursor.getString(0),
                        "timeframe" to cursor.getString(1),
                        "timestamp" to cursor.getString(2),
                        "close"     to cursor.getString(3)
                    )
                )
            }
            cursor.close()

            assertEquals(
                "All pre-migration candle rows must be present after v1→v2 upgrade",
                2, rows.size
            )
            assertTrue(
                "Every pre-migration candle must be assigned timeframe='daily'",
                rows.all { it["timeframe"] == "daily" }
            )
            assertEquals("VOO", rows[0]["ticker"])
            assertEquals("2024-01-02", rows[0]["timestamp"])
            assertEquals("VOO", rows[1]["ticker"])
            assertEquals("2024-01-03", rows[1]["timestamp"])
            assertEquals("gap_up", run {
                val c2 = db.openHelper.readableDatabase.query(
                    "SELECT gap_type FROM candles WHERE timestamp='2024-01-03'"
                )
                c2.moveToFirst()
                val v = c2.getString(0)
                c2.close()
                v
            })
        } finally {
            db.close()
        }
    }

    /**
     * Creates a v2-shaped database, inserts two signal_history rows (which have no conviction
     * columns at v2), runs MIGRATION_2_3, and confirms:
     *   - both rows are still present
     *   - conviction_tier and conviction_reasons are NULL for every pre-existing row
     *
     * This guards against a future DROP/recreate mistake in MIGRATION_2_3 that would
     * silently wipe saved signals instead of just adding the new nullable columns.
     */
    @Test
    fun migration2To3_preservesExistingSignalHistoryRowsWithNullConvictionTier() {
        val dbName2 = "signal_migration_2_3_test.db"
        context.deleteDatabase(dbName2)

        // ── Step 1: build a hand-crafted v2 database ──────────────────────────
        val dbFile = context.getDatabasePath(dbName2)
        dbFile.parentFile?.mkdirs()

        val v2 = SQLiteDatabase.openOrCreateDatabase(dbFile, null)
        v2.execSQL("PRAGMA user_version = 2")

        // v2 candles (with timeframe column) — required for Room schema validation
        v2.execSQL(
            """
            CREATE TABLE IF NOT EXISTS candles (
                ticker               TEXT    NOT NULL,
                timeframe            TEXT    NOT NULL DEFAULT 'daily',
                timestamp            TEXT    NOT NULL,
                open                 REAL    NOT NULL,
                high                 REAL    NOT NULL,
                low                  REAL    NOT NULL,
                close                REAL    NOT NULL,
                volume               INTEGER NOT NULL,
                is_extended_hours    INTEGER NOT NULL,
                session_type         TEXT    NOT NULL,
                gap_percent          REAL,
                gap_type             TEXT,
                PRIMARY KEY(ticker, timeframe, timestamp)
            )
            """.trimIndent()
        )

        // v2 signal_history: no conviction columns yet
        v2.execSQL(
            """
            CREATE TABLE IF NOT EXISTS signal_history (
                id                       TEXT    NOT NULL,
                timestamp                TEXT    NOT NULL,
                ticker                   TEXT    NOT NULL,
                cycle_id                 TEXT,
                signal_type              TEXT    NOT NULL,
                gauge_type               TEXT    NOT NULL,
                confidence               REAL    NOT NULL,
                session_type             TEXT    NOT NULL,
                is_extended_hours        INTEGER NOT NULL,
                gap_type                 TEXT,
                liquidity_score          REAL    NOT NULL,
                macro_override_applied   INTEGER NOT NULL,
                PRIMARY KEY(id)
            )
            """.trimIndent()
        )

        // confidence_history is unchanged
        v2.execSQL(
            """
            CREATE TABLE IF NOT EXISTS confidence_history (
                id                       TEXT    NOT NULL,
                timestamp                TEXT    NOT NULL,
                ticker                   TEXT    NOT NULL,
                long_buy_confidence      REAL    NOT NULL,
                long_sell_confidence     REAL    NOT NULL,
                short_buy_confidence     REAL    NOT NULL,
                short_sell_confidence    REAL    NOT NULL,
                session_type             TEXT    NOT NULL,
                is_extended_hours        INTEGER NOT NULL,
                PRIMARY KEY(id)
            )
            """.trimIndent()
        )

        // Insert two pre-existing signal rows (no conviction data at v2)
        v2.execSQL(
            """
            INSERT INTO signal_history VALUES
                ('sig-001','2024-03-01T10:00:00','VOO',NULL,'BUY','long_trend',0.72,'regular',0,NULL,1.0,0)
            """.trimIndent()
        )
        v2.execSQL(
            """
            INSERT INTO signal_history VALUES
                ('sig-002','2024-03-02T11:30:00','VOO','cycle-42','SELL','short_trend',0.65,'regular',0,'gap_up',0.9,1)
            """.trimIndent()
        )
        v2.close()

        // ── Step 2: open with Room, applying only MIGRATION_2_3 ───────────────
        val db = Room.databaseBuilder(
            context,
            NovaCycleDatabase::class.java,
            dbName2
        )
            .addMigrations(AppModule.MIGRATION_2_3)
            .build()

        try {
            // ── Step 3: verify all signal rows survived ────────────────────────
            val cursor = db.openHelper.readableDatabase.query(
                "SELECT id, signal_type, conviction_tier, conviction_reasons " +
                        "FROM signal_history ORDER BY timestamp"
            )

            val rows = mutableListOf<Map<String, String?>>()
            while (cursor.moveToNext()) {
                rows.add(
                    mapOf(
                        "id"                 to cursor.getString(0),
                        "signal_type"        to cursor.getString(1),
                        "conviction_tier"    to if (cursor.isNull(2)) null else cursor.getString(2),
                        "conviction_reasons" to if (cursor.isNull(3)) null else cursor.getString(3)
                    )
                )
            }
            cursor.close()

            assertEquals(
                "Both pre-migration signal rows must still exist after v2→v3 upgrade",
                2, rows.size
            )
            assertTrue(
                "conviction_tier must be NULL for every pre-existing signal row",
                rows.all { it["conviction_tier"] == null }
            )
            assertTrue(
                "conviction_reasons must be NULL for every pre-existing signal row",
                rows.all { it["conviction_reasons"] == null }
            )
            assertEquals("sig-001", rows[0]["id"])
            assertEquals("BUY", rows[0]["signal_type"])
            assertEquals("sig-002", rows[1]["id"])
            assertEquals("SELL", rows[1]["signal_type"])
        } finally {
            db.close()
            context.deleteDatabase(dbName2)
        }
    }

    /**
     * Creates a v2-shaped database, inserts two confidence_history rows, runs MIGRATION_2_3,
     * and asserts that both rows are still present with every field value intact.
     *
     * MIGRATION_2_3 only touches signal_history (adds conviction columns), so
     * confidence_history must be completely unaffected. This test guards against a future
     * migration that accidentally DROPs or recreates confidence_history as a side-effect.
     */
    @Test
    fun migration2To3_preservesConfidenceHistoryRows() {
        val dbName3 = "confidence_migration_2_3_test.db"
        context.deleteDatabase(dbName3)

        // ── Step 1: build a hand-crafted v2 database ──────────────────────────
        val dbFile = context.getDatabasePath(dbName3)
        dbFile.parentFile?.mkdirs()

        val v2 = SQLiteDatabase.openOrCreateDatabase(dbFile, null)
        v2.execSQL("PRAGMA user_version = 2")

        // v2 candles (with timeframe column) — required for Room schema validation
        v2.execSQL(
            """
            CREATE TABLE IF NOT EXISTS candles (
                ticker               TEXT    NOT NULL,
                timeframe            TEXT    NOT NULL DEFAULT 'daily',
                timestamp            TEXT    NOT NULL,
                open                 REAL    NOT NULL,
                high                 REAL    NOT NULL,
                low                  REAL    NOT NULL,
                close                REAL    NOT NULL,
                volume               INTEGER NOT NULL,
                is_extended_hours    INTEGER NOT NULL,
                session_type         TEXT    NOT NULL,
                gap_percent          REAL,
                gap_type             TEXT,
                PRIMARY KEY(ticker, timeframe, timestamp)
            )
            """.trimIndent()
        )

        // v2 signal_history — required for Room schema validation
        v2.execSQL(
            """
            CREATE TABLE IF NOT EXISTS signal_history (
                id                       TEXT    NOT NULL,
                timestamp                TEXT    NOT NULL,
                ticker                   TEXT    NOT NULL,
                cycle_id                 TEXT,
                signal_type              TEXT    NOT NULL,
                gauge_type               TEXT    NOT NULL,
                confidence               REAL    NOT NULL,
                session_type             TEXT    NOT NULL,
                is_extended_hours        INTEGER NOT NULL,
                gap_type                 TEXT,
                liquidity_score          REAL    NOT NULL,
                macro_override_applied   INTEGER NOT NULL,
                PRIMARY KEY(id)
            )
            """.trimIndent()
        )

        // confidence_history is unchanged across all versions
        v2.execSQL(
            """
            CREATE TABLE IF NOT EXISTS confidence_history (
                id                       TEXT    NOT NULL,
                timestamp                TEXT    NOT NULL,
                ticker                   TEXT    NOT NULL,
                long_buy_confidence      REAL    NOT NULL,
                long_sell_confidence     REAL    NOT NULL,
                short_buy_confidence     REAL    NOT NULL,
                short_sell_confidence    REAL    NOT NULL,
                session_type             TEXT    NOT NULL,
                is_extended_hours        INTEGER NOT NULL,
                PRIMARY KEY(id)
            )
            """.trimIndent()
        )

        // Insert two confidence_history rows that must survive the migration
        v2.execSQL(
            """
            INSERT INTO confidence_history VALUES
                ('ch-001','2024-05-01T09:30:00','VOO',0.82,0.18,0.45,0.55,'regular',0)
            """.trimIndent()
        )
        v2.execSQL(
            """
            INSERT INTO confidence_history VALUES
                ('ch-002','2024-05-02T09:30:00','VOO',0.61,0.39,0.70,0.30,'regular',0)
            """.trimIndent()
        )
        v2.close()

        // ── Step 2: open with Room, applying only MIGRATION_2_3 ───────────────
        val db = Room.databaseBuilder(
            context,
            NovaCycleDatabase::class.java,
            dbName3
        )
            .addMigrations(AppModule.MIGRATION_2_3)
            .build()

        try {
            // ── Step 3: verify confidence_history rows survived intact ─────────
            val cursor = db.openHelper.readableDatabase.query(
                "SELECT id, ticker, long_buy_confidence, long_sell_confidence, " +
                        "short_buy_confidence, short_sell_confidence, session_type, is_extended_hours " +
                        "FROM confidence_history ORDER BY timestamp"
            )

            val rows = mutableListOf<Map<String, String>>()
            while (cursor.moveToNext()) {
                rows.add(
                    mapOf(
                        "id"                   to cursor.getString(0),
                        "ticker"               to cursor.getString(1),
                        "long_buy_confidence"  to cursor.getString(2),
                        "long_sell_confidence" to cursor.getString(3),
                        "short_buy_confidence" to cursor.getString(4),
                        "short_sell_confidence"to cursor.getString(5),
                        "session_type"         to cursor.getString(6),
                        "is_extended_hours"    to cursor.getString(7)
                    )
                )
            }
            cursor.close()

            assertEquals(
                "Both confidence_history rows must still be present after v2→v3 migration",
                2, rows.size
            )

            // First row assertions
            assertEquals("ch-001", rows[0]["id"])
            assertEquals("VOO", rows[0]["ticker"])
            assertEquals("0.82", rows[0]["long_buy_confidence"])
            assertEquals("0.18", rows[0]["long_sell_confidence"])
            assertEquals("0.45", rows[0]["short_buy_confidence"])
            assertEquals("0.55", rows[0]["short_sell_confidence"])
            assertEquals("regular", rows[0]["session_type"])
            assertEquals("0", rows[0]["is_extended_hours"])

            // Second row assertions
            assertEquals("ch-002", rows[1]["id"])
            assertEquals("VOO", rows[1]["ticker"])
            assertEquals("0.61", rows[1]["long_buy_confidence"])
            assertEquals("0.39", rows[1]["long_sell_confidence"])

            // Verify second row's remaining confidence values via a targeted query
            val c2 = db.openHelper.readableDatabase.query(
                "SELECT long_sell_confidence, short_buy_confidence, short_sell_confidence " +
                        "FROM confidence_history WHERE id='ch-002'"
            )
            c2.moveToFirst()
            val longSell2  = c2.getFloat(0)
            val shortBuy2  = c2.getFloat(1)
            val shortSell2 = c2.getFloat(2)
            c2.close()

            assertEquals("long_sell_confidence for ch-002",  0.39f, longSell2,  0.001f)
            assertEquals("short_buy_confidence for ch-002",  0.70f, shortBuy2,  0.001f)
            assertEquals("short_sell_confidence for ch-002", 0.30f, shortSell2, 0.001f)
        } finally {
            db.close()
            context.deleteDatabase(dbName3)
        }
    }
}
