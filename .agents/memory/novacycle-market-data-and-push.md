---
name: NovaCycle market data and push readiness
description: Vendor OHLC validation and the two-part FCM readiness requirement
---

Reject vendor OHLC rows before persistence when high is below open/close, low is above open/close, values are non-positive, non-finite, or non-numeric. Re-read the latest boundary timestamp so corrected vendor data can repair an older bad row, and clean invalid persisted rows before prediction startup.

**Why:** A contradictory Yahoo Finance daily candle made VOO appear bearish even though intraday data showed a rise. A valid model signal still cannot reach a phone unless both Firebase server credentials and an Android-registered FCM token exist.

**How to apply:** Keep `/api/healthz` notification readiness secret-free, reporting only whether FCM is configured, registered-device count, and blockers. For real delivery, configure the Android Firebase project and backend service-account secret, then install a new APK and confirm a token appears.

For chart explanations, keep the freshest positive-volume VOO price separate from the exact model-input prices: long-trend uses the latest valid daily close, while short-trend uses the latest OHLC-valid 5-minute close, including zero-volume bars retained for feature computation.

**Why:** A user-facing “current price” and a model’s feature-input price can legitimately differ by timeframe or by data-quality handling; combining them makes a recommendation appear to use the wrong market price.

**How to apply:** Any chart or recommendation explainer should label current/latest, long-model, and short-model prices independently and include their timestamps when space allows.