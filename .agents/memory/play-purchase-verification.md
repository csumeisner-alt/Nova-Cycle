---
name: Play purchase verification design
description: How Mint Luxe entitlement is verified server-side and the fail-open/fail-closed rules
---

# Server-side Play purchase verification

The backend verifies Play purchase tokens via the Play Developer API
(`purchases.products.get`) using a service account (secret
`PLAY_SERVICE_ACCOUNT_JSON`, package from `PLAY_PACKAGE_NAME`, falls back to
`FCM_SERVER_KEY` if that service account also has androidpublisher scope).

**Rules:**
- Server verdict is authoritative when it answers: entitled → unlock; invalid/refunded → lock (even if the local Play client claims ownership).
- Server unreachable (network, 5xx, 503 unconfigured) → fail-open: honour the Play-client-verified purchase provisionally and re-verify on next app start. Never fail-closed on infrastructure errors or real buyers lose purchases offline.
- Refunds are detected on the entitlement re-check endpoint (called during purchase restore), which re-queries Play every time.

**Why:** rooted devices can edit SharedPreferences / spoof local billing state but cannot forge a valid purchase token; the token must round-trip through Google.

**How to apply:** any new paid entitlement should reuse this verify/recheck flow and the same fail-open-on-unreachable, fail-closed-on-rejection decision logic (pure logic lives in the Android domain layer with unit tests).
