# Nova Cycle — Android Build Guide

This is the full Kotlin/Jetpack Compose Android app for NovaCycle.  
Build it with Android Studio, the Gradle command line, or **automatically with GitHub Actions** — **no Expo, no React Native, no Firebase account required**.

---

## Requirements

| Tool | Version |
|------|---------|
| Android Studio | Hedgehog (2023.1.1) or newer |
| JDK | 17 |
| Android SDK | 35 (compileSdk), min SDK 26 |
| Gradle | Bundled via `gradlew` wrapper |

---

## Fastest option: Download the APK from GitHub Actions (no setup needed)

We provide an automated GitHub Actions workflow that builds the APK for you every time code is pushed.

### How to download the APK

1. **Get the code on GitHub** — ask your agent to push it to a GitHub repo (or create one and push manually).
2. Open the repo on GitHub.
3. Go to the **Actions** tab → click **Build NovaCycle APK** on the left.
4. Click the latest successful run.
5. Scroll to the **Artifacts** section and click **novacyle-debug-apk** to download `app-debug.apk`.
6. Transfer the APK to your Android phone and install it.

> **Note:** Push notifications are disabled in this build because they require a Firebase project. The rest of the app — gauges, charts, indicators, reliability, settings — works fully without it.

---

## Option 2: Build locally with Android Studio

1. Open `android/` as the root in Android Studio (File → Open → select the folder containing `settings.gradle.kts`).
2. Wait for Gradle sync.
3. **Build → Build Bundle(s) / APK(s) → Build APK(s)**.
4. The APK appears at:
   ```
   android/app/build/outputs/apk/debug/app-debug.apk
   ```

---

## Option 3: Build from the command line

```bash
cd android
./gradlew assembleDebug
```

The APK is written to:

```
android/app/build/outputs/apk/debug/app-debug.apk
```

---

## Install the APK on your device

**Via ADB (USB debugging enabled):**

```bash
adb install android/app/build/outputs/apk/debug/app-debug.apk
```

**Via file transfer:**  
Copy the APK to your phone and open it. You may need to allow **Install unknown apps** for your file manager in Settings → Security.

---

## About Firebase / Push Notifications

- **Not required to build or use the app.**
- **To enable real BUY/SELL push notifications later:**
  1. Create a Firebase project at https://console.firebase.google.com
  2. Add an Android app with package name `com.novacycle`
  3. Download `google-services.json` and place it at `android/app/google-services.json`
  4. Un-comment the three Firebase-related lines in `android/app/build.gradle.kts`
  5. Un-comment the FCM `<service>` block in `android/app/src/main/AndroidManifest.xml`
  6. Replace the stub at `android/app/src/main/kotlin/com/novacycle/notifications/NovaCycleFirebaseService.kt` with the full implementation (the original code is in the comments)
  7. Set the Firebase service account JSON as the `FCM_SERVER_KEY` secret on the Replit backend

---

## Backend URL

The app is configured to talk to the live Replit backend in `android/app/build.gradle.kts`:

```kotlin
buildConfigField("String", "API_BASE_URL", "\"https://YOUR_REPLIT_DOMAIN/api/\"")
```

If your Replit domain changes, update this URL and rebuild (or push, and GitHub Actions will rebuild the APK).

---

## Package name

`com.novacycle` — used by Firebase and the Play Store if you publish later.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `google-services.json not found` | Not needed for the debug build. See Firebase section above only if you want push notifications. |
| `Connection refused` on real device | Update `API_BASE_URL` in `android/app/build.gradle.kts` to your current Replit URL. |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | Uninstall any previous version of the app first. |
| Gradle sync fails on JDK version | Ensure JDK 17 is selected in Android Studio → Settings → Build Tools → Gradle. |
