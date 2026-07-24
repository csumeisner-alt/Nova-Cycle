"""
NovaCycle – Notifications Router
=================================
Endpoints for managing FCM device tokens and triggering test notifications.

Routes (all prefixed with /api by main.py):
  POST   /register_device      — store or refresh an FCM token
  DELETE /unregister_device    — remove an FCM token
  GET    /device_tokens        — list registered tokens (debug)
  POST   /test_notification    — send a test push to all tokens (debug)
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_session
from database.models import DeviceToken

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class RegisterDeviceRequest(BaseModel):
    token: str
    device_name: Optional[str] = None


class TestNotificationRequest(BaseModel):
    signal_type: str = "buy"   # 'buy' | 'sell'
    gauge_type: str = "long"   # 'long' | 'short'
    confidence: float = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/register_device")
async def register_device(
    body: RegisterDeviceRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Register or refresh an FCM device token.

    If the token already exists, its device_name and updated_at are refreshed.
    If it's new, a row is inserted.  Idempotent — safe to call on every app launch.
    """
    if not body.token or len(body.token) < 10:
        raise HTTPException(status_code=400, detail="A valid FCM token is required")

    result = await session.execute(
        select(DeviceToken).where(DeviceToken.token == body.token)
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.device_name = body.device_name
        existing.updated_at = datetime.utcnow()
        logger.info("FCM token refreshed for device: %s", body.device_name or "unknown")
    else:
        session.add(DeviceToken(
            token=body.token,
            device_name=body.device_name,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ))
        logger.info("FCM token registered for device: %s", body.device_name or "unknown")

    await session.commit()
    return {"status": "ok", "message": "Device token registered"}


@router.delete("/unregister_device")
async def unregister_device(
    token: str,
    session: AsyncSession = Depends(get_session),
):
    """Remove an FCM device token. Safe to call even if token doesn't exist."""
    await session.execute(delete(DeviceToken).where(DeviceToken.token == token))
    await session.commit()
    logger.info("FCM token unregistered: %s...", token[:20])
    return {"status": "ok", "message": "Device token removed"}


@router.get("/device_tokens")
async def list_device_tokens(session: AsyncSession = Depends(get_session)):
    """List all registered device tokens (tokens are truncated for safety)."""
    result = await session.execute(select(DeviceToken).order_by(DeviceToken.updated_at.desc()))
    tokens = result.scalars().all()
    return [
        {
            "id": t.id,
            "token_preview": t.token[:20] + "...",
            "device_name": t.device_name,
            "created_at": t.created_at.isoformat(),
            "updated_at": t.updated_at.isoformat(),
        }
        for t in tokens
    ]


@router.post("/test_notification")
async def test_notification(
    body: TestNotificationRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Send a test push notification to all registered devices.
    Useful for verifying end-to-end Firebase setup without waiting for a real signal.
    """
    from notifications.fcm import FCMNotifier

    result = await session.execute(select(DeviceToken))
    tokens = result.scalars().all()

    if not tokens:
        raise HTTPException(
            status_code=404,
            detail="No device tokens registered. Open the app on your phone first."
        )

    notifier = FCMNotifier()
    results = []
    for device in tokens:
        ok = await notifier.send_signal_notification(
            device_token=device.token,
            signal_type=body.signal_type,
            gauge_type=body.gauge_type,
            confidence=body.confidence,
            is_extended=False,
            score=75.0,
            gap_type="none",
        )
        results.append({
            "device_name": device.device_name or "unknown",
            "token_preview": device.token[:20] + "...",
            "sent": ok,
        })

    return {"status": "ok", "results": results}
