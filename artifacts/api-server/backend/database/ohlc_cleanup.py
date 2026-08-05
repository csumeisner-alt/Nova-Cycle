"""
OHLC Malformed-Candle Cleanup
=============================
One-time (and on-demand) retroactive removal of candles that violate internal
OHLC consistency rules (e.g. high < open, low > close).

These rows slipped in before the ingest-time validation was in place.  They
are harmless while prediction-time filtering catches them, but they inflate the
healthz ohlc_quarantine counter on every restart and add noise to the feature
pipeline.  Deleting them once is safer than accumulating them indefinitely.

Public API
----------
    summary = await remove_malformed_candles(session)

Returns a dict:
    {
        "rows_found":          int,   # total violating rows across all tables
        "rows_removed":        int,   # rows actually deleted
        "tables_affected":     list[str],
        "timeframes_affected": list[str],
        "details":             list[dict],  # per-table breakdown
    }

The caller is responsible for committing the session (or rolling back on error).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    VooCandle, SpxCandle, VixCandle,
    VixShortCandle, VixLongCandle, RatesCandle,
    CreditHyCandle, CreditIgCandle, BreadthCandle,
)
from ingestion.ohlc_validator import validate_ohlc_row

logger = logging.getLogger(__name__)

# Tables to scan: (model_class, human_name)
_CANDLE_TABLES: list[tuple[Any, str]] = [
    (VooCandle, "voo_candles"),
    (VixCandle, "vix_candles"),
    (SpxCandle, "spx_candles"),
    # Broader market context tables
    (VixShortCandle, "vix_short_candles"),
    (VixLongCandle, "vix_long_candles"),
    (RatesCandle, "rates_candles"),
    (CreditHyCandle, "credit_hy_candles"),
    (CreditIgCandle, "credit_ig_candles"),
    (BreadthCandle, "breadth_candles"),
]


async def remove_malformed_candles(session: AsyncSession) -> dict:
    """Scan all candle tables and delete rows that violate OHLC constraints.

    Loads rows in batches of 2 000 to avoid building a huge in-memory list.
    Idempotent: safe to call multiple times (already-clean rows are never
    touched).

    Args:
        session: An open async SQLAlchemy session.  The caller must call
                 ``await session.commit()`` after this function returns.

    Returns:
        Summary dict (see module docstring).
    """
    total_found = 0
    total_removed = 0
    tables_affected: list[str] = []
    timeframes_affected_set: set[str] = set()
    details: list[dict] = []

    for model_cls, table_name in _CANDLE_TABLES:
        found = 0
        removed = 0
        bad_ids: list[int] = []
        timeframe_set: set[str] = set()

        # Stream rows in chunks
        BATCH = 2_000
        offset = 0
        while True:
            result = await session.execute(
                select(model_cls).offset(offset).limit(BATCH)
            )
            rows = result.scalars().all()
            if not rows:
                break

            for row in rows:
                ok, reason = validate_ohlc_row(
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                )
                if not ok:
                    bad_ids.append(row.id)
                    tf = getattr(row, "timeframe", "unknown")
                    timeframe_set.add(tf)
                    logger.debug(
                        "ohlc_cleanup found malformed row table=%s id=%d ts=%s reason=%s",
                        table_name, row.id,
                        getattr(row, "timestamp", "?"), reason,
                    )

            offset += BATCH
            if len(rows) < BATCH:
                break

        found = len(bad_ids)

        # Single bulk DELETE — one atomic round-trip, no per-row fetching.
        # Either every bad row is removed or none are (rolled back by caller).
        if bad_ids:
            result = await session.execute(
                delete(model_cls).where(model_cls.id.in_(bad_ids))
            )
            removed = result.rowcount

        if found:
            tables_affected.append(table_name)
            timeframes_affected_set.update(timeframe_set)
            logger.info(
                "ohlc_cleanup table=%s rows_found=%d rows_removed=%d timeframes=%s",
                table_name, found, removed, sorted(timeframe_set),
            )

        details.append({
            "table": table_name,
            "rows_found": found,
            "rows_removed": removed,
            "timeframes": sorted(timeframe_set),
        })

        total_found += found
        total_removed += removed

    summary = {
        "rows_found": total_found,
        "rows_removed": total_removed,
        "tables_affected": tables_affected,
        "timeframes_affected": sorted(timeframes_affected_set),
        "details": details,
    }

    logger.info(
        "ohlc_cleanup complete rows_found=%d rows_removed=%d tables=%s timeframes=%s",
        total_found, total_removed, tables_affected, sorted(timeframes_affected_set),
    )
    return summary
