"""
NovaCycle Firebase Cloud Messaging (FCM) Notifier
===================================================
Sends push notifications via the FCM HTTP v1 API using a service account.

FCM_SERVER_KEY must contain the full JSON content of a Firebase service account
key file (Firebase Console → Project Settings → Service Accounts → Generate new
private key). The JSON is stored as a single-line string in the Replit Secret.

Gracefully no-ops when FCM_SERVER_KEY is not configured.
"""

import json
import logging
import asyncio
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

FCM_SCOPES = ["https://www.googleapis.com/auth/firebase.messaging"]


def _get_access_token(service_account_json: str) -> tuple[str, str]:
    """
    Exchange service account credentials for an OAuth2 bearer token.
    Returns (access_token, project_id).

    Runs synchronously — always call via asyncio.run_in_executor.
    """
    from google.oauth2 import service_account
    import google.auth.transport.requests

    info = json.loads(service_account_json)
    project_id = info.get("project_id", "")
    if not project_id:
        raise ValueError("Service account JSON is missing 'project_id'")

    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=FCM_SCOPES
    )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token, project_id


class FCMNotifier:
    """Send FCM push notifications for NovaCycle trading signals (HTTP v1 API)."""

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
        conviction_tier: Optional[str] = None,
    ) -> bool:
        """
        Send a BUY/SELL signal push notification to a specific device.

        Args:
            device_token:    FCM device registration token.
            signal_type:     'buy' or 'sell'
            gauge_type:      'long' or 'short'
            confidence:      float in [0, 1]
            is_extended:     True if this is an extended-hours signal
            score:           Raw gauge score (optional)
            gap_type:        'gap_up', 'gap_down', or 'none' (optional)
            liquidity_score: float (optional)

        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        if not settings.FCM_SERVER_KEY:
            logger.info(
                "FCM_SERVER_KEY not configured — skipping notification (%s %s signal)",
                gauge_type, signal_type,
            )
            return False

        if not device_token:
            logger.warning("No device token provided — skipping FCM notification")
            return False

        auth = await self._get_auth()
        if not auth:
            return False
        access_token, project_id = auth

        title, body = self._build_message(
            signal_type, gauge_type, confidence, is_extended, gap_type,
            conviction_tier,
        )

        payload = {
            "message": {
                "token": device_token,
                "notification": {
                    "title": title,
                    "body": body,
                },
                "data": {
                    "signal_type": signal_type,
                    "gauge_type": gauge_type,
                    "confidence": str(round(confidence, 4)),
                    "is_extended_hours": str(is_extended).lower(),
                    "score": str(round(score, 2)) if score is not None else "",
                    "gap_type": gap_type or "none",
                    "liquidity_score": (
                        str(round(liquidity_score, 4))
                        if liquidity_score is not None else ""
                    ),
                    "ticker": settings.TICKER,
                    "conviction_tier": conviction_tier or "",
                },
                "android": {
                    "priority": "HIGH",
                    "notification": {"sound": "default"},
                },
                "apns": {
                    "payload": {"aps": {"sound": "default"}},
                },
            }
        }

        return await self._post(payload, access_token, project_id)

    async def send_confidence_momentum_alert(
        self,
        device_token: str,
        gauge_type: str,
        old_confidence: float,
        new_confidence: float,
        direction: str,
    ) -> bool:
        """
        Send an alert when confidence is changing rapidly.

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

        auth = await self._get_auth()
        if not auth:
            return False
        access_token, project_id = auth

        change_pct = abs(new_confidence - old_confidence) * 100.0
        title = f"NovaCycle – {gauge_type.capitalize()} Confidence {direction.capitalize()}"
        body = (
            f"Confidence moved {change_pct:.1f}% {direction} "
            f"({old_confidence:.0%} → {new_confidence:.0%})"
        )

        payload = {
            "message": {
                "token": device_token,
                "notification": {"title": title, "body": body},
                "data": {
                    "alert_type": "confidence_momentum",
                    "gauge_type": gauge_type,
                    "old_confidence": str(round(old_confidence, 4)),
                    "new_confidence": str(round(new_confidence, 4)),
                    "direction": direction,
                    "ticker": settings.TICKER,
                },
                "android": {"priority": "NORMAL"},
            }
        }

        return await self._post(payload, access_token, project_id)

    async def send_training_stuck_alert(
        self,
        device_token: str,
        model_name: str,
        consecutive_failures: int,
        last_error: Optional[str] = None,
    ) -> bool:
        """
        Send an operator alert when a model's retraining is stuck (repeated
        consecutive failures crossed the alert threshold).

        Returns:
            True on success.
        """
        if not settings.FCM_SERVER_KEY:
            logger.info(
                "FCM_SERVER_KEY not configured — skipping training-stuck alert (%s)",
                model_name,
            )
            return False

        if not device_token:
            logger.warning("No device token provided — skipping training-stuck alert")
            return False

        auth = await self._get_auth()
        if not auth:
            return False
        access_token, project_id = auth

        pretty_name = model_name.replace("_", " ").title()
        title = f"⚠️ NovaCycle – {pretty_name} Retraining Stuck"
        body = (
            f"{pretty_name} model failed {consecutive_failures} consecutive "
            f"retrain attempts. Predictions may be degraded."
        )
        if last_error:
            body += f" Last error: {str(last_error)[:120]}"

        payload = {
            "message": {
                "token": device_token,
                "notification": {"title": title, "body": body},
                "data": {
                    "alert_type": "training_stuck",
                    "model_name": model_name,
                    "consecutive_failures": str(int(consecutive_failures)),
                    "ticker": settings.TICKER,
                },
                "android": {
                    "priority": "HIGH",
                    "notification": {"sound": "default"},
                },
                "apns": {
                    "payload": {"aps": {"sound": "default"}},
                },
            }
        }

        return await self._post(payload, access_token, project_id)

    async def send_baseline_duration_alert(
        self,
        device_token: str,
        model_name: str,
        days_in_baseline: float,
        threshold_days: float,
    ) -> bool:
        """Send an operator alert when the long-trend model has stayed in
        baseline mode (no gate-passing trained model) past the configured
        threshold.

        Returns:
            True on success.
        """
        if not settings.FCM_SERVER_KEY:
            logger.info(
                "FCM_SERVER_KEY not configured — skipping baseline-duration alert (%s)",
                model_name,
            )
            return False

        if not device_token:
            logger.warning("No device token provided — skipping baseline-duration alert")
            return False

        auth = await self._get_auth()
        if not auth:
            return False
        access_token, project_id = auth

        pretty_name = model_name.replace("_", " ").title()
        days_int = int(days_in_baseline)
        title = f"⚠️ NovaCycle – {pretty_name} Stuck in Baseline Mode"
        body = (
            f"{pretty_name} has been serving the calibrated base rate for "
            f"{days_int} day(s) (threshold: {int(threshold_days)}d). "
            f"No retrain has passed the OOS quality gate. "
            f"Investigate features or data quality."
        )

        payload = {
            "message": {
                "token": device_token,
                "notification": {"title": title, "body": body},
                "data": {
                    "alert_type": "baseline_mode_duration",
                    "model_name": model_name,
                    "days_in_baseline": str(days_int),
                    "threshold_days": str(int(threshold_days)),
                    "ticker": settings.TICKER,
                },
                "android": {
                    "priority": "HIGH",
                    "notification": {"sound": "default"},
                },
                "apns": {
                    "payload": {"aps": {"sound": "default"}},
                },
            }
        }

        return await self._post(payload, access_token, project_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _get_auth(self) -> Optional[tuple[str, str]]:
        """Obtain (access_token, project_id) using the service account JSON."""
        try:
            loop = asyncio.get_event_loop()
            token, project_id = await loop.run_in_executor(
                None, _get_access_token, settings.FCM_SERVER_KEY
            )
            return token, project_id
        except Exception as exc:
            logger.error("Failed to obtain FCM access token: %s", exc)
            return None

    @staticmethod
    def _build_message(
        signal_type: str,
        gauge_type: str,
        confidence: float,
        is_extended: bool,
        gap_type: Optional[str],
        conviction_tier: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build a human-readable notification title and body."""
        emoji_map = {"buy": "🟢", "sell": "🔴", "neutral": "⚪"}
        emoji = emoji_map.get(signal_type.lower(), "📊")

        session_tag = " [Extended Hours]" if is_extended else ""
        horizon = "Long-Term" if gauge_type == "long" else "Short-Term"
        action = signal_type.upper()
        tier_tag = " ⭐" if conviction_tier == "high_conviction" else ""

        title = f"{emoji} NovaCycle {horizon} {action}{session_tag}{tier_tag}"

        body_parts = [
            f"VOO {horizon} {action} signal",
            f"Confidence: {confidence:.0%}",
        ]
        if conviction_tier == "high_conviction":
            body_parts.insert(0, "High-Conviction")
        if gap_type and gap_type != "none":
            body_parts.append(f"Gap: {gap_type.replace('_', ' ').title()}")
        if is_extended:
            body_parts.append("⚠️ Extended hours – lower liquidity")

        return title, " | ".join(body_parts)

    async def _post(self, payload: dict, access_token: str, project_id: str) -> bool:
        """POST the FCM v1 payload to the correct project endpoint."""
        url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    logger.info("FCM notification sent successfully")
                    return True
                else:
                    logger.error(
                        "FCM HTTP error %d: %s",
                        response.status_code,
                        response.text[:300],
                    )
                    return False
        except httpx.TimeoutException:
            logger.error("FCM request timed out")
            return False
        except Exception as exc:
            logger.error("FCM send error: %s", exc)
            return False
