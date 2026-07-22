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
    """Create all ORM-defined tables (no-op if they already exist)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


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
