"""
NovaCycle DB Maintenance
========================
One-time / startup repair routines.

`reclassify_session_labels` re-runs the DST/holiday-aware
`market_calendar.classify_session` over stored 5-minute candles and fixes
any rows whose `session_type` / `is_extended_hours` labels differ from
what the current classifier produces (e.g. candles ingested before the
calendar-aware classifier existed).

The pass is idempotent: a second run finds nothing to correct.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import VooCandle
from ingestion.market_calendar import classify_session

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000


async def reclassify_session_labels(
    session: AsyncSession, timeframe: str = "5min"
) -> int:
    """
    Re-classify stored candles of the given timeframe and update rows whose
    `session_type` / `is_extended_hours` differ from the calendar-aware
    classifier. Returns the number of corrected rows. Idempotent.
    """
    scanned = 0
    corrected = 0
    last_id = 0

    while True:
        result = await session.execute(
            select(VooCandle)
            .where(VooCandle.timeframe == timeframe, VooCandle.id > last_id)
            .order_by(VooCandle.id)
            .limit(_BATCH_SIZE)
        )
        rows = result.scalars().all()
        if not rows:
            break

        for candle in rows:
            last_id = candle.id
            scanned += 1
            is_ext, session_type, _method = classify_session(candle.timestamp)
            if (
                candle.session_type != session_type
                or bool(candle.is_extended_hours) != is_ext
            ):
                candle.session_type = session_type
                candle.is_extended_hours = is_ext
                corrected += 1

        await session.flush()

    await session.commit()
    logger.info(
        "session_label_reclassification timeframe=%s scanned=%d corrected=%d",
        timeframe,
        scanned,
        corrected,
    )
    return corrected
