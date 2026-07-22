"""
NovaCycle History Router
=========================
GET /api/model_metadata?ticker=VOO
  Returns model training history from the ModelMetadata table.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.db import get_db
from database.models import ModelMetadata

logger = logging.getLogger(__name__)

router = APIRouter(tags=["history"])


def _validate_ticker(ticker: str) -> str:
    if ticker.upper() != settings.TICKER.upper():
        raise HTTPException(
            status_code=400,
            detail=f"Multi-ticker support not yet available. Only '{settings.TICKER}' is supported.",
        )
    return ticker.upper()


@router.get("/model_metadata")
async def get_model_metadata(
    ticker: str = Query(default="VOO", description="Ticker symbol (only VOO supported)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return all model training metadata records for the given ticker,
    ordered by most-recent first.

    Response fields per record:
      - id, model_name, ticker, trained_at
      - accuracy (float)
      - feature_importances (dict, parsed from JSON)
    """
    _validate_ticker(ticker)

    try:
        result = await db.execute(
            select(ModelMetadata)
            .where(ModelMetadata.ticker == settings.TICKER)
            .order_by(ModelMetadata.trained_at.desc())
        )
        rows = result.scalars().all()

        records = []
        for r in rows:
            fi = {}
            if r.feature_importances:
                try:
                    fi = json.loads(r.feature_importances)
                except Exception:
                    fi = {}
            records.append(
                {
                    "id": r.id,
                    "model_name": r.model_name,
                    "ticker": r.ticker,
                    "trained_at": r.trained_at.isoformat() if r.trained_at else None,
                    "accuracy": r.accuracy,
                    "feature_importances": fi,
                }
            )

        return {"ticker": ticker, "count": len(records), "records": records}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_model_metadata error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
