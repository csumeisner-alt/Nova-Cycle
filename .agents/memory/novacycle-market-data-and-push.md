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

## VIX index volume semantics
Treat `^VIX` volume as non-trading metadata: Yahoo Finance returns valid VIX OHLC candles with volume `0` (and sometimes `NULL`). VIX ingestion and prediction reads must validate OHLC, but must not reject rows solely for non-positive volume. Keep zero-volume rejection for traded feeds such as VOO and ES futures.

**Why:** `^VIX` is an index rather than a traded security. Applying the generic zero-volume filter emptied the VIX table, left macro predictions degraded, and prevented recovery because incremental ingestion skipped an empty table.

**How to apply:** Use an explicit VIX exception in fetch normalization and storage. Still surface missing/stale VIX data when the vendor fetch is empty or fails, and preserve neutral/fallback model behavior in that case.

## Visible live-price source
The displayed VOO quote must prefer Yahoo's session-specific live quote (`preMarketPrice`, `regularMarketPrice`, or `postMarketPrice`) over persisted candles. Persisted 5-minute candles remain the fallback and the model-input prices remain separate.

**Why:** Yahoo extended-hours history can lag or omit the latest quote even while `postMarketPrice` is available, causing the UI to show the regular close as “live.”

**How to apply:** Keep live quote fetching bounded and failure-safe, never substitute `previousClose`, and expose the source/session so the UI can explain whether the value is live or candle-backed.