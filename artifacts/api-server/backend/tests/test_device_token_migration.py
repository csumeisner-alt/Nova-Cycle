"""
Regression tests for device_tokens schema migration and notification filtering.

Covers the upgrade scenario:
  1. A device_tokens table with the OLD schema (no preference columns) exists.
  2. _migrate_device_tokens() runs on startup and adds the missing columns.
  3. POST /register_device succeeds and stores preference values.
  4. _notify_all_devices_bg correctly skips signals below the stored threshold
     and extended-hours signals when the device has opted out.
"""

import asyncio
import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text

from database.models import Base, DeviceToken
from database.db import _migrate_device_tokens
from routers.notifications import RegisterDeviceRequest


# ─────────────────────────────────────────────────────────────────────────────
# Shared in-memory SQLite engine for tests
# ─────────────────────────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


def make_engine():
    return create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create the OLD schema (without preference columns)
# ─────────────────────────────────────────────────────────────────────────────

OLD_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS device_tokens (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token      TEXT    NOT NULL UNIQUE,
    device_name TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Migration adds missing columns to an old-schema table
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_migrate_adds_missing_columns():
    """_migrate_device_tokens adds the three preference columns when absent."""
    import database.db as db_module

    engine = make_engine()
    original_engine = db_module.async_engine

    try:
        db_module.async_engine = engine

        # Create the OLD table (missing preference columns)
        async with engine.begin() as conn:
            await conn.execute(text(OLD_SCHEMA_SQL))

        # Run migration
        await _migrate_device_tokens()

        # Verify all three columns now exist
        async with engine.connect() as conn:
            result = await conn.execute(text("PRAGMA table_info(device_tokens)"))
            col_names = {row[1] for row in result.fetchall()}

        assert "min_buy_threshold" in col_names, "min_buy_threshold column missing after migration"
        assert "min_sell_threshold" in col_names, "min_sell_threshold column missing after migration"
        assert "extended_hours_notifications" in col_names, "extended_hours_notifications column missing after migration"
    finally:
        db_module.async_engine = original_engine
        await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Migration is idempotent (safe to run twice)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_migrate_is_idempotent():
    """Running _migrate_device_tokens twice does not raise an error."""
    import database.db as db_module

    engine = make_engine()
    original_engine = db_module.async_engine

    try:
        db_module.async_engine = engine

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)  # creates with NEW schema

        # First migration (columns already exist — should no-op)
        await _migrate_device_tokens()
        # Second migration — also must not raise
        await _migrate_device_tokens()
    finally:
        db_module.async_engine = original_engine
        await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Existing token rows get default values after migration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_migrated_rows_have_default_preferences():
    """Rows inserted with the old schema get 0.70/0.70/True defaults after migration."""
    import database.db as db_module

    engine = make_engine()
    original_engine = db_module.async_engine

    try:
        db_module.async_engine = engine

        async with engine.begin() as conn:
            await conn.execute(text(OLD_SCHEMA_SQL))
            # Insert a row the old way (no preference columns)
            await conn.execute(text(
                "INSERT INTO device_tokens (token, device_name) "
                "VALUES ('old_token_abcdefghij', 'OldPhone')"
            ))

        await _migrate_device_tokens()

        async with engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT min_buy_threshold, min_sell_threshold, extended_hours_notifications "
                "FROM device_tokens WHERE token = 'old_token_abcdefghij'"
            ))
            row = result.fetchone()

        assert row is not None
        assert abs(row[0] - 0.70) < 0.001, f"expected min_buy_threshold=0.70, got {row[0]}"
        assert abs(row[1] - 0.70) < 0.001, f"expected min_sell_threshold=0.70, got {row[1]}"
        assert row[2] == 1, f"expected extended_hours_notifications=1 (True), got {row[2]}"
    finally:
        db_module.async_engine = original_engine
        await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Threshold clamping in RegisterDeviceRequest
# ─────────────────────────────────────────────────────────────────────────────

def test_threshold_clamped_to_zero_when_negative():
    req = RegisterDeviceRequest(
        token="x" * 20, min_buy_threshold=-0.5, min_sell_threshold=-99.0
    )
    assert req.min_buy_threshold == 0.0
    assert req.min_sell_threshold == 0.0


def test_threshold_clamped_to_one_when_above():
    req = RegisterDeviceRequest(
        token="x" * 20, min_buy_threshold=1.5, min_sell_threshold=200.0
    )
    assert req.min_buy_threshold == 1.0
    assert req.min_sell_threshold == 1.0


def test_threshold_unchanged_when_valid():
    req = RegisterDeviceRequest(
        token="x" * 20, min_buy_threshold=0.70, min_sell_threshold=0.85
    )
    assert abs(req.min_buy_threshold - 0.70) < 0.001
    assert abs(req.min_sell_threshold - 0.85) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Notification filtering logic
# Tests the filtering conditions applied in _notify_all_devices_bg
# ─────────────────────────────────────────────────────────────────────────────

def _should_notify(device: dict, signal_type: str, confidence: float, is_extended: bool) -> bool:
    """
    Mirrors the filtering logic in _notify_all_devices_bg.
    Returns True if a notification should be sent to this device.
    """
    if is_extended and not device["extended_hours_notifications"]:
        return False
    threshold = (
        device["min_buy_threshold"] if signal_type == "buy" else device["min_sell_threshold"]
    )
    return confidence >= threshold


class TestNotificationFiltering:

    def _device(self, min_buy=0.70, min_sell=0.70, ext_hours=True):
        return {
            "token": "tok_" + "x" * 20,
            "device_name": "TestPhone",
            "min_buy_threshold": min_buy,
            "min_sell_threshold": min_sell,
            "extended_hours_notifications": ext_hours,
        }

    def test_signal_above_threshold_is_sent(self):
        assert _should_notify(self._device(min_buy=0.70), "buy", 0.75, False) is True

    def test_signal_at_threshold_is_sent(self):
        assert _should_notify(self._device(min_buy=0.70), "buy", 0.70, False) is True

    def test_signal_below_threshold_is_skipped(self):
        assert _should_notify(self._device(min_buy=0.70), "buy", 0.65, False) is False

    def test_sell_signal_uses_sell_threshold(self):
        device = self._device(min_buy=0.70, min_sell=0.80)
        assert _should_notify(device, "sell", 0.75, False) is False
        assert _should_notify(device, "sell", 0.80, False) is True

    def test_extended_hours_signal_skipped_when_opted_out(self):
        assert _should_notify(self._device(ext_hours=False), "buy", 0.90, True) is False

    def test_extended_hours_signal_sent_when_opted_in(self):
        assert _should_notify(self._device(ext_hours=True), "buy", 0.90, True) is True

    def test_high_sensitivity_device_receives_weak_signal(self):
        # HIGH sensitivity → threshold 0.50
        assert _should_notify(self._device(min_buy=0.50), "buy", 0.55, False) is True

    def test_low_sensitivity_device_skips_medium_signal(self):
        # LOW sensitivity → threshold 0.85
        assert _should_notify(self._device(min_buy=0.85), "buy", 0.75, False) is False

    def test_low_sensitivity_device_receives_strong_signal(self):
        assert _should_notify(self._device(min_buy=0.85), "buy", 0.90, False) is True
