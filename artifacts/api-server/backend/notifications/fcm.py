"""
NovaCycle Firebase Cloud Messaging (FCM) Notifier
===================================================
Sends push notifications via the legacy FCM HTTP v1 API.

Notification types:
  - Long-term BUY / SELL signals
  - Short-term BUY / SELL signals
  - Extended-hours signal alerts
  - Confidence momentum alerts

Gracefully no-ops when FCM_SERVER_KEY is not configured.
"""

import logging
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

FCM_ENDPOINT = "https://fcm.googleapis.com/fcm/send"


class FCMNotifier:
    """Send FCM push notifications for NovaCycle trading signals."""

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def send_signal_notification(
        self,
        device_token: str,
        signal_type: str,
        gauge_type: str,
        confidence: float,
        is_extended: bool,
        score: Optional[float] = None,
        gap_type: Optional[str] = None,
        liquidity_score: Optional[float] = None,
    ) -> bool:
        """
        Send a signal push notification to a specific device.

        Args:
            device_token:    FCM device registration token.
            signal_type:     'buy' or 'sell'
            gauge_type:      'long' or 'short'
            confidence:      float in [0, 1]
            is_extended:     True if this is an extended-hours signal
            score:           Raw gauge score (optional, for data payload)
            gap_type:        'gap_up', 'gap_down', or 'none' (optional)
            liquidity_score: float (optional)

        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        if not settings.FCM_SERVER_KEY:
            logger.info(
                "FCM_SERVER_KEY not configured — skipping notification "
                "(%s %s signal)", gauge_type, signal_type
            )
            return False

        if not device_token:
            logger.warning("No device token provided — skipping FCM notification")
            return False

        title, body = self._build_message(
            signal_type, gauge_type, confidence, is_extended, gap_type
        )

        payload = {
            "to": device_token,
            "notification": {
                "title": title,
                "body": body,
                "sound": "default",
            },
            "data": {
                "signal_type": signal_type,
                "gauge_type": gauge_type,
                "confidence": str(round(confidence, 4)),
                "is_extended_hours": str(is_extended).lower(),
                "score": str(round(score, 2)) if score is not None else "",
                "gap_type": gap_type or "none",
                "liquidity_score": str(round(liquidity_score, 4)) if liquidity_score is not None else "",
                "ticker": settings.TICKER,
            },
            "priority": "high",
        }

        return await self._post(payload)

    async def send_confidence_momentum_alert(
        self,
        device_token: str,
        gauge_type: str,
        old_confidence: float,
        new_confidence: float,
        direction: str,
    ) -> bool:
        """
        Send an alert when confidence is changing rapidly (momentum alert).

        Args:
            device_token:    FCM device token
            gauge_type:      'long' or 'short'
            old_confidence:  Previous confidence value
            new_confidence:  Current confidence value
            direction:       'rising' or 'falling'

        Returns:
            True on success.
        """
        if not settings.FCM_SERVER_KEY:
            return False

        change_pct = abs(new_confidence - old_confidence) * 100.0
        title = f"NovaCycle – {gauge_type.capitalize()} Confidence {direction.capitalize()}"
        body = (
            f"Confidence moved {change_pct:.1f}% {direction} "
            f"({old_confidence:.0%} → {new_confidence:.0%})"
        )

        payload = {
            "to": device_token,
            "notification": {"title": title, "body": body, "sound": "default"},
            "data": {
                "alert_type": "confidence_momentum",
                "gauge_type": gauge_type,
                "old_confidence": str(round(old_confidence, 4)),
                "new_confidence": str(round(new_confidence, 4)),
                "direction": direction,
                "ticker": settings.TICKER,
            },
            "priority": "normal",
        }

        return await self._post(payload)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_message(
        signal_type: str,
        gauge_type: str,
        confidence: float,
        is_extended: bool,
        gap_type: Optional[str],
    ) -> tuple[str, str]:
        """
        Build a human-readable notification title and body.

        Returns:
            (title: str, body: str)
        """
        emoji_map = {"buy": "🟢", "sell": "🔴", "neutral": "⚪"}
        emoji = emoji_map.get(signal_type.lower(), "📊")

        session_tag = " [Extended Hours]" if is_extended else ""
        horizon = "Long-Term" if gauge_type == "long" else "Short-Term"
        action = signal_type.upper()

        title = f"{emoji} NovaCycle {horizon} {action}{session_tag}"

        body_parts = [
            f"VOO {horizon} {action} signal",
            f"Confidence: {confidence:.0%}",
        ]
        if gap_type and gap_type != "none":
            body_parts.append(f"Gap: {gap_type.replace('_', ' ').title()}")
        if is_extended:
            body_parts.append("⚠️ Extended hours – lower liquidity")

        body = " | ".join(body_parts)
        return title, body

    async def _post(self, payload: dict) -> bool:
        """
        POST the FCM payload to the FCM endpoint.

        Returns:
            True if HTTP 200 received, False on any error.
        """
        headers = {
            "Authorization": f"key={settings.FCM_SERVER_KEY}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    FCM_ENDPOINT,
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 200:
                    resp_json = response.json()
                    if resp_json.get("failure", 0) > 0:
                        logger.warning(
                            "FCM reported delivery failure: %s", resp_json
                        )
                        return False
                    logger.info("FCM notification sent successfully")
                    return True
                else:
                    logger.error(
                        "FCM HTTP error %d: %s",
                        response.status_code,
                        response.text[:200],
                    )
                    return False
        except httpx.TimeoutException:
            logger.error("FCM request timed out")
            return False
        except Exception as exc:
            logger.error("FCM send error: %s", exc)
            return False
