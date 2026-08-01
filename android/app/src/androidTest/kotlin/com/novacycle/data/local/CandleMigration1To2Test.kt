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
}
