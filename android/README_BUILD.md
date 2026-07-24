# Nova Cycle — Android Build Guide

This is the full Kotlin/Jetpack Compose Android app for NovaCycle.  
Build it with Android Studio or the Gradle command line — **no Expo, no React Native**.

---

## Requirements

| Tool | Version |
|------|---------|
| Android Studio | Hedgehog (2023.1.1) or newer |
| JDK | 17 |
| Android SDK | 35 (compileSdk), min SDK 26 |
| Gradle | Bundled via `gradlew` wrapper |

---

## Step 1 — Add google-services.json

Firebase is required for push notifications (BUY/SELL alerts).

1. Go to [https://console.firebase.google.com/](https://console.firebase.google.com/)
2. Create or select a project
3. Click **Add app → Android**, package name: `com.novacycle`
4. Download `google-services.json`
5. Place it at `android/app/google-services.json`

> **Without this file the build will fail.** The placeholder at  
> `android/app/google-services.json.placeholder` describes what to do.

If you want to build without Firebase for now, see **Building without Firebase** below.

---

## Step 2 — Set the API base URL (if needed)

The URL is pre-set to the live Replit backend in `android/app/build.gradle.kts`:

```kotlin
buildConfigField("String", "API_BASE_URL", "\"https://<your-replit-domain>/api/\"")
```

- **Real device → Replit backend**: keep the `https://` URL as-is.  
- **Emulator → local machine**: change to `"http://10.0.2.2:8080/api/"`.  
- **Real device → local machine**: use your machine's LAN IP, e.g. `"http://192.168.1.x:8080/api/"`.

---

## Step 3 — Build the debug APK

### Option A — Android Studio (recommended)

1. Open Android Studio
2. **File → Open** → select the `android/` folder (the one containing `settings.gradle.kts`)
3. Wait for Gradle sync to complete
4. **Build → Build Bundle(s) / APK(s) → Build APK(s)**
5. Click **locate** in the notification to find the APK

### Option B — Command line

```bash
cd android
./gradlew assembleDebug
```

The output APK is at:

```
android/app/build/outputs/apk/debug/app-debug.apk
```

---

## Step 4 — Install on your device

**Via ADB (USB debugging enabled on device):**

```bash
adb install android/app/build/outputs/apk/debug/app-debug.apk
```

**Via file transfer:**  
Copy `app-debug.apk` to your device and open it. You may need to enable  
**Settings → Security → Install unknown apps** for your file manager.

---

## Building without Firebase

To build without a `google-services.json` (disables push notifications):

1. In `android/app/build.gradle.kts`, comment out these two lines:
   ```kotlin
   // alias(libs.plugins.google.services)    ← in the plugins block
   // implementation(libs.firebase.messaging) ← in dependencies
   ```
2. In `android/app/src/main/kotlin/com/novacycle/notifications/NovaCycleFirebaseService.kt`,
   comment out the class body or delete the file.
3. Remove the Firebase service entry from `AndroidManifest.xml`.

All other screens (gauges, charts, indicators, reliability) work without Firebase.

---

## App Screens

| Screen | What it does |
|--------|-------------|
| **Dual Gauge** | Spring-animated long/short trend gauges, auto-refreshes every 5 min |
| **Raw Chart** | Zoomable candlestick chart with 7 signal marker types, tap for Story Card |
| **Filtered Chart** | Strongest-confidence signals only, trade-cycle shading, confidence ribbon |
| **Confidence History** | Long + short confidence curves with EMA smoothing |
| **Indicators** | RSI, Stochastic, MACD, Bollinger, ADX, VIX regime, and more |
| **Hold Time** | AI-estimated position duration with reasoning bullets |
| **Settings** | Thresholds, weighting mode, smoothing, notification sensitivity |
| **Reliability** | BUY→SELL cycle win-rate metrics, sortable/filterable table |

---

## Package name

`com.novacycle` — used in Firebase Console and the Play Store if you publish.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `google-services.json not found` | Add the file to `android/app/` (Step 1) |
| `Connection refused` on real device | Update `API_BASE_URL` to the Replit HTTPS URL (Step 2) |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | Uninstall any previous version of the app first |
| Gradle sync fails on JDK version | Ensure JDK 17 is selected in Android Studio → Settings → Build Tools → Gradle |
