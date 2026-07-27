---
name: Android API model synchronization
description: Keep Android Moshi response models in sync with backend JSON shape; strict codegen adapters throw on unknown keys or type mismatches.
---

When the NovaCycle backend adds new response fields or changes a field type, the Android app may crash on launch even if the backend tests pass. The Android client uses Moshi with `@JsonClass(generateAdapter = true)` for most response models; generated adapters are strict by default and raise `JsonDataException` when they encounter unknown JSON keys or values that do not match the declared Kotlin type.

**Rule:** after any backend change that affects `/predict_long`, `/predict_short`, `/indicators`, `/healthz`, or `/hold_time_estimate`, update the matching Android models in `android/app/src/main/kotlin/com/novacycle/data/remote/models/` before cutting a release APK. Add new fields with sensible defaults and, when a field may contain mixed types (e.g. `indicator_breakdown` with numeric scores plus textual annotations), widen the model type to `Map<String, Any>` and filter to numeric values in the UI.

**Why:** on app launch, `DualGaugeViewModel` calls `predict_long`, `predict_short`, `hold_time_estimate`, and `indicators` in parallel, while `HealthViewModel` polls `/healthz`. Any parsing failure in these startup paths prevents the dashboard from loading. The repository wraps calls in `runCatching`, but if the exception is unhandled in a different path (or if the user perceives a blank/error screen as a crash), the app becomes unusable.

**How to apply:**
1. Check the actual backend JSON response for each endpoint, not just the declared Python return type.
2. Add missing fields to the Kotlin data class with default values so Moshi does not fail on unknown keys.
3. When a field type changes (e.g. `reasoning` from `String` to `List<String>`), update the model and any UI that consumes it.
4. Run the CI `Build NovaCycle APK` workflow on the merged `main` branch before telling the user the APK is ready; the rolling `latest` release is the installable artifact.
