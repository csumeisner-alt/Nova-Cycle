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

from database.models import Base

# ─────────────────────────────────────────────────────────────────────────────
# Engine — always use SQLite (aiosqlite).
# We hardcode this URL to avoid Replit's DATABASE_URL env var (PostgreSQL)
# overriding the pydantic-settings default.
# ─────────────────────────────────────────────────────────────────────────────
_SQLITE_URL = "sqlite+aiosqlite:///./novacycle.db"
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


async def _migrate_device_tokens() -> None:
    """
    Add the three notification-preference columns to an existing device_tokens
    table that was created before they were introduced.

    SQLite does not support `ALTER TABLE … ADD COLUMN IF NOT EXISTS`, so we
    check the column list first and only issue the ALTER when missing.
    This is idempotent and safe to run on every startup.
    """
    _NEW_COLS = {
        "min_buy_threshold":           "REAL NOT NULL DEFAULT 0.70",
        "min_sell_threshold":          "REAL NOT NULL DEFAULT 0.70",
        "extended_hours_notifications": "INTEGER NOT NULL DEFAULT 1",
    }

    async with async_engine.connect() as conn:
        # Fetch current column names from PRAGMA
        result = await conn.execute(
            __import__("sqlalchemy").text("PRAGMA table_info(device_tokens)")
        )
        existing_cols = {row[1] for row in result.fetchall()}  # row[1] = column name

        for col_name, col_def in _NEW_COLS.items():
            if col_name not in existing_cols:
                await conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE device_tokens ADD COLUMN {col_name} {col_def}"
                    )
                )
                import logging
                logging.getLogger(__name__).info(
                    "Migrated device_tokens: added column %s", col_name
                )
        await conn.commit()


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
