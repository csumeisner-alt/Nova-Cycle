"""
NovaCycle Database Engine
=========================
Async SQLAlchemy engine (SQLite via aiosqlite).
Provides:
  - async_engine  – SQLAlchemy async engine
  - AsyncSessionLocal – session factory
  - init_db()     – create all tables on startup
  - get_db()      – FastAPI dependency (async context manager)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import logging as _logging
import os as _os

from database.models import Base

# ─────────────────────────────────────────────────────────────────────────────
# Engine — always use SQLite (aiosqlite).
# We hardcode this URL to avoid Replit's DATABASE_URL env var (PostgreSQL)
# overriding the pydantic-settings default.
# ─────────────────────────────────────────────────────────────────────────────
_SQLITE_URL = "sqlite+aiosqlite:///./novacycle.db"

# Warn loudly when DATABASE_URL is set in the environment but ignored.
# In production (Replit Reserved VM) DATABASE_URL points at PostgreSQL, but
# NovaCycle stores all data in the local SQLite file.  Without this warning
# operators may spend hours debugging "wrong database" failures.
_env_db_url = _os.environ.get("DATABASE_URL", "")
if _env_db_url and "sqlite" not in _env_db_url.lower():
    # Extract only the scheme and host for logging — never log the full URL
    # which may contain embedded credentials (user:password@host).
    try:
        from urllib.parse import urlparse as _urlparse
        _parsed = _urlparse(_env_db_url)
        _safe_url = f"{_parsed.scheme}://{_parsed.hostname}"
    except Exception:
        _safe_url = "<non-sqlite>"
    _logging.getLogger(__name__).warning(
        "db_url_ignored DATABASE_URL points at %s (non-SQLite) but NovaCycle "
        "always uses the local SQLite file (%s).  The env var is intentionally "
        "ignored — remove it from production config to avoid confusion.",
        _safe_url,
        _SQLITE_URL,
    )

async_engine = create_async_engine(
    _SQLITE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session factory
# expire_on_commit=False keeps ORM objects usable after commit
# ─────────────────────────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Table initialisation
# ─────────────────────────────────────────────────────────────────────────────
async def init_db() -> None:
    """
    Create all ORM-defined tables (no-op if they already exist), then run
    lightweight column-backfill migrations for tables that predate new columns.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _migrate_device_tokens()
    await _migrate_conviction_columns()
    await _migrate_model_state_column()


async def _add_missing_columns(table: str, new_cols: dict) -> None:
    """
    Add any missing columns to an existing table.

    SQLite does not support `ALTER TABLE … ADD COLUMN IF NOT EXISTS`, so we
    check the column list first and only issue the ALTER when missing.
    This is idempotent and safe to run on every startup.
    """
    from sqlalchemy import text
    import logging

    async with async_engine.connect() as conn:
        result = await conn.execute(text(f"PRAGMA table_info({table})"))
        existing_cols = {row[1] for row in result.fetchall()}  # row[1] = column name

        for col_name, col_def in new_cols.items():
            if col_name not in existing_cols:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
                )
                logging.getLogger(__name__).info(
                    "Migrated %s: added column %s", table, col_name
                )
        await conn.commit()


async def _migrate_device_tokens() -> None:
    """Backfill notification-preference columns on device_tokens."""
    await _add_missing_columns("device_tokens", {
        "min_buy_threshold":            "REAL NOT NULL DEFAULT 0.70",
        "min_sell_threshold":           "REAL NOT NULL DEFAULT 0.70",
        "extended_hours_notifications": "INTEGER NOT NULL DEFAULT 1",
        "high_conviction_only":         "INTEGER NOT NULL DEFAULT 0",
    })


async def _migrate_conviction_columns() -> None:
    """Backfill conviction-tier columns on signal tables (NULL = pre-tiering)."""
    await _add_missing_columns("signal_history", {
        "conviction_tier":    "VARCHAR(24)",
        "conviction_reasons": "TEXT",
    })
    await _add_missing_columns("filtered_signals", {
        "conviction_tier": "VARCHAR(24)",
    })


async def _migrate_model_state_column() -> None:
    """Backfill the model_state column on signal_history (NULL = unknown/pre-column)."""
    await _add_missing_columns("signal_history", {
        "model_state": "VARCHAR(32)",
    })


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency
# ─────────────────────────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async DB session, ensuring rollback on error and close on exit.

    Usage::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Aliases used by main.py for compatibility
# ─────────────────────────────────────────────────────────────────────────────
create_tables = init_db          # async callable → creates all tables

def get_session_factory():
    """Return the async session factory (compatible with `async with factory() as session`)."""
    return AsyncSessionLocal


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency alias for get_db (used by routers)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context-manager version for use outside of FastAPI dependency injection
    (e.g. from the scheduler or startup hooks).

    Usage::

        async with get_db_context() as db:
            result = await db.execute(...)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
