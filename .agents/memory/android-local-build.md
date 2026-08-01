---
name: Android local build fallback
description: When GitHub delivery is unavailable, build and attach a directly installable NovaCycle APK from the Repl.
---

The Android app can be built locally when GitHub Actions is unavailable, but the Repl may not have an Android SDK or a standard JDK. Bootstrap the SDK under the path referenced by `android/local.properties`, use a standard Temurin JDK 17 for Android Gradle Plugin packaging, and build the debug variant when CI release-signing credentials are unavailable.

**Why:** The CI release keystore is not available in the Repl, while the debug APK can still be verified and delivered directly as an installable artifact.

**How to apply:** Treat the debug APK as a fresh install if an older CI-signed APK is present; Android may reject it as an update because the signing keys differ. Warn that uninstalling the older package removes its local app data.