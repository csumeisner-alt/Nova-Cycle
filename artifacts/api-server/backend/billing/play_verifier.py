"""
Google Play purchase verification
=================================
Verifies in-app purchase tokens against the Google Play Developer API
(`purchases.products.get`) using a service account.

Configuration (Replit Secrets / env):
  - PLAY_SERVICE_ACCOUNT_JSON — full JSON of a Google Cloud service account
    that has been granted access to the Play Console (Financial data /
    "View app information" permissions). Falls back to FCM_SERVER_KEY if the
    same service account is used for both (it must additionally have the
    androidpublisher scope granted in Play Console).
  - PLAY_PACKAGE_NAME — Android applicationId (default: com.novacycle).

Fails explicitly: raises VerifierNotConfigured when no credentials are set,
and PlayApiError on Play API/network failures. No silent fallbacks.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

ANDROIDPUBLISHER_SCOPES = ["https://www.googleapis.com/auth/androidpublisher"]

# purchaseState values from the Play Developer API
PURCHASE_STATE_PURCHASED = 0
PURCHASE_STATE_CANCELED = 1
PURCHASE_STATE_PENDING = 2


class VerifierNotConfigured(Exception):
    """No Play service account credentials are configured."""


class PlayApiError(Exception):
    """The Play Developer API could not be reached or returned a server error."""


class InvalidPurchaseToken(Exception):
    """Play rejected the token — it does not correspond to a real purchase."""


@dataclass
class PlayPurchase:
    """Normalized result of purchases.products.get."""
    purchase_state: int          # 0 purchased, 1 cancelled/refunded, 2 pending
    order_id: Optional[str]
    acknowledged: bool

    @property
    def is_entitled(self) -> bool:
        return self.purchase_state == PURCHASE_STATE_PURCHASED


def _service_account_json() -> str:
    raw = settings.PLAY_SERVICE_ACCOUNT_JSON or settings.FCM_SERVER_KEY
    if not raw:
        raise VerifierNotConfigured(
            "PLAY_SERVICE_ACCOUNT_JSON is not configured — cannot verify purchases"
        )
    return raw


def _get_access_token(service_account_json: str) -> str:
    """Sync — run via run_in_executor."""
    from google.oauth2 import service_account
    import google.auth.transport.requests

    info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=ANDROIDPUBLISHER_SCOPES
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


class PlayPurchaseVerifier:
    """Verifies purchase tokens via the Play Developer API."""

    async def verify(self, product_id: str, purchase_token: str) -> PlayPurchase:
        """
        Look up a purchase token on Google Play.

        Raises:
            VerifierNotConfigured — no service account configured.
            InvalidPurchaseToken  — Play returned 4xx (token is not real).
            PlayApiError          — network failure or Play 5xx.
        """
        sa_json = _service_account_json()
        loop = asyncio.get_running_loop()
        try:
            token = await loop.run_in_executor(None, _get_access_token, sa_json)
        except Exception as exc:  # bad JSON, auth failure, network
            raise PlayApiError(f"Could not obtain Play API access token: {exc}") from exc

        url = (
            "https://androidpublisher.googleapis.com/androidpublisher/v3/applications/"
            f"{settings.PLAY_PACKAGE_NAME}/purchases/products/{product_id}/tokens/{purchase_token}"
        )
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            raise PlayApiError(f"Play API request failed: {exc}") from exc

        if resp.status_code >= 500:
            raise PlayApiError(f"Play API server error {resp.status_code}")
        if resp.status_code >= 400:
            # 400/404 = token or product does not exist; 401/403 = bad credentials
            if resp.status_code in (401, 403):
                raise PlayApiError(
                    f"Play API rejected our credentials ({resp.status_code}) — "
                    "check the service account's Play Console access"
                )
            logger.warning(
                "Play rejected purchase token for %s: HTTP %s", product_id, resp.status_code
            )
            raise InvalidPurchaseToken(f"Play API returned {resp.status_code} for token")

        data = resp.json()
        return PlayPurchase(
            purchase_state=int(data.get("purchaseState", PURCHASE_STATE_CANCELED)),
            order_id=data.get("orderId"),
            acknowledged=int(data.get("acknowledgementState", 0)) == 1,
        )


# Module-level default instance; FastAPI dependency indirection lets tests
# substitute a fake.
_verifier = PlayPurchaseVerifier()


def get_play_verifier() -> PlayPurchaseVerifier:
    return _verifier
