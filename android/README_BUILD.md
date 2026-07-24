# Nova Cycle — Android Build Guide

This is the full Kotlin/Jetpack Compose Android app for NovaCycle.  
Build it with Android Studio or the Gradle command line — **no Expo, no React Native**.

---

## Quick start — build the APK right now

Firebase push notifications are **disabled by default** so you can build without any
Firebase setup. All 8 screens and live data from the Replit backend work immediately.

```bash
cd android
./gradlew assembleDebug
```

The output APK is at:

```
android/app/build/outputs/apk/debug/app-debug.apk
```

Install it:

```bash
# Via ADB (USB debugging on, device connected):
adb install android/app/build/outputs/apk/debug/app-debug.apk

# Or copy app-debug.apk to your device and open it.
# You may need: Settings → Security → Install unknown apps
```

> **First build is slow** (~5–10 min) while Gradle downloads dependencies.  
> Subsequent incremental builds take 30–60 seconds.

---

## Requirements

| Tool | Version |
|------|---------|
| Android Studio | Hedgehog (2023.1.1) or newer |
| JDK | 17 |
| Android SDK | 35 (compileSdk), min SDK 26 (Android 8+) |
| Gradle | Bundled via `gradlew` wrapper |

---

## Step 1 — Set the API base URL (if needed)

The URL is pre-set to the live Replit backend in `android/app/build.gradle.kts`:

```kotlin
buildConfigField("String", "API_BASE_URL", "\"https://<your-replit-domain>/api/\"")
```

- **Real device → Replit backend**: keep the `https://` URL as-is.  
- **Emulator → local machine**: change to `"http://10.0.2.2:8080/api/"`.  
- **Real device → local machine**: use your machine's LAN IP, e.g. `"http://192.168.1.x:8080/api/"`.

---

## Step 2 — Build via Android Studio (alternative)

1. **File → Open** → select the `android/` folder (the one containing `settings.gradle.kts`)
2. Wait for Gradle sync to complete
3. **Build → Build Bundle(s) / APK(s) → Build APK(s)**
4. Click **locate** in the notification to find `app-debug.apk`

---

## Enabling Firebase push notifications (optional)

Push notifications (BUY/SELL alerts) require a Firebase project. To enable them:

### 1 — Create the Firebase config

1. Go to [https://console.firebase.google.com/](https://console.firebase.google.com/)
2. Create or select a project
3. Click **Add app → Android**, package name: `com.novacycle`
4. Download `google-services.json`
5. Place it at `android/app/google-services.json`

### 2 — Re-enable Firebase in Gradle

In `android/app/build.gradle.kts`, un-comment these three lines:

```kotlin
// In the plugins block:
alias(libs.plugins.google.services)   // ← un-comment

// In dependencies:
implementation(platform(libs.firebase.bom))    // ← un-comment
implementation(libs.firebase.messaging)        // ← un-comment
```

### 3 — Re-enable the FCM service in the manifest

In `android/app/src/main/AndroidManifest.xml`, un-comment the service block:

```xml
<service
    android:name=".notifications.NovaCycleFirebaseService"
    android:exported="false">
    <intent-filter>
        <action android:name="com.google.firebase.MESSAGING_EVENT" />
    </intent-filter>
</service>
```

### 4 — Restore NovaCycleFirebaseService and NovaCycleApp

In `NovaCycleFirebaseService.kt`, replace the stub `object` with the full `class`
implementation shown in the file's comments.

In `NovaCycleApp.kt`, restore the Firebase import and the `fetchFcmToken()` call
(also shown in comments in that file).

---

## App screens

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
| `Connection refused` on real device | Update `API_BASE_URL` to the Replit HTTPS URL in `build.gradle.kts` |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | Uninstall any previous version of the app first |
| Gradle sync fails on JDK version | Ensure JDK 17 is selected in Android Studio → Settings → Build Tools → Gradle |
| `google-services.json not found` | Either add the file (see Enabling Firebase above) or keep Firebase commented out |
| Build succeeds but no push notifications | Firebase is disabled — see Enabling Firebase above |
