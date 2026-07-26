"""
Billing Router
==============
Server-side verification of Google Play in-app purchases (Mint Luxe theme).

The Android client must call POST /api/billing/verify_purchase with the Play
purchase token before unlocking Mint Luxe locally, and calls
GET /api/billing/entitlement on later app starts to re-check the token —
which is where server-side refund detection revokes the entitlement.

Explicit failure modes (no silent fallbacks):
  - 503 when no Play service account is configured
  - 502 when the Play API is unreachable / errors
  - 200 {entitled: false, ...} when Play says the token is fake, pending,
    cancelled, or refunded
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_session
from database.models import PurchaseEntitlement
from billing.play_verifier import (
    InvalidPurchaseToken,
    PlayApiError,
    PlayPurchaseVerifier,
    VerifierNotConfigured,
    get_play_verifier,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class VerifyPurchaseRequest(BaseModel):
    product_id: str = Field(min_length=1, max_length=128)
    purchase_token: str = Field(min_length=1, max_length=1024)


class EntitlementResponse(BaseModel):
    entitled: bool
    state: str          # 'active' | 'revoked' | 'pending' | 'invalid'
    product_id: str


async def _verify_and_record(
    product_id: str,
    purchase_token: str,
    session: AsyncSession,
    verifier: PlayPurchaseVerifier,
) -> EntitlementResponse:
    """Verify a token against Play and upsert the entitlement row."""
    try:
        play = await verifier.verify(product_id, purchase_token)
    except VerifierNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except PlayApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except InvalidPurchaseToken:
        # Fake / non-existent token — never entitle, record nothing.
        return EntitlementResponse(entitled=False, state="invalid", product_id=product_id)

    result = await session.execute(
        select(PurchaseEntitlement).where(
            PurchaseEntitlement.purchase_token == purchase_token
        )
    )
    row = result.scalar_one_or_none()
    now = datetime.utcnow()

    if play.is_entitled:
        if row is None:
            session.add(PurchaseEntitlement(
                product_id=product_id,
                purchase_token=purchase_token,
                order_id=play.order_id,
                state="active",
                play_purchase_state=play.purchase_state,
                verified_at=now,
                last_checked_at=now,
            ))
            logger.info("Verified new %s purchase (order %s)", product_id, play.order_id)
        else:
            if row.state == "revoked":
                # Play says purchased again (e.g. re-bought after refund).
                logger.info("Re-activating previously revoked %s entitlement", product_id)
                row.revoked_at = None
            row.state = "active"
            row.play_purchase_state = play.purchase_state
            row.order_id = play.order_id or row.order_id
            row.last_checked_at = now
        await session.commit()
        return EntitlementResponse(entitled=True, state="active", product_id=product_id)

    # Not entitled: pending, cancelled, or refunded
    if play.purchase_state == 2:
        state = "pending"
    else:
        state = "revoked"

    if row is not None:
        row.play_purchase_state = play.purchase_state
        row.last_checked_at = now
        if state == "revoked" and row.state != "revoked":
            logger.info(
                "Revoking %s entitlement — Play reports purchaseState=%s (refund/cancel)",
                product_id, play.purchase_state,
            )
            row.state = "revoked"
            row.revoked_at = now
        await session.commit()

    return EntitlementResponse(entitled=False, state=state, product_id=product_id)


@router.post("/billing/verify_purchase", response_model=EntitlementResponse)
async def verify_purchase(
    body: VerifyPurchaseRequest,
    session: AsyncSession = Depends(get_session),
    verifier: PlayPurchaseVerifier = Depends(get_play_verifier),
):
    """Verify a fresh purchase token against Google Play and record the entitlement."""
    return await _verify_and_record(body.product_id, body.purchase_token, session, verifier)


@router.get("/billing/entitlement", response_model=EntitlementResponse)
async def check_entitlement(
    product_id: str,
    purchase_token: str,
    session: AsyncSession = Depends(get_session),
    verifier: PlayPurchaseVerifier = Depends(get_play_verifier),
):
    """
    Re-verify a previously seen purchase token.

    Always re-queries Play, so refunds issued since the original purchase are
    detected here and the stored entitlement flips to 'revoked'.
    """
    return await _verify_and_record(product_id, purchase_token, session, verifier)
