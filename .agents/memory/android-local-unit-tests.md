---
name: Android local unit tests on Replit
description: How to run the Android app's JVM unit tests in the Replit workspace (no preinstalled java/SDK)
---

# Running Android unit tests locally

No `java` on PATH and no Android SDK by default, but gradle unit tests CAN run:

```bash
export JAVA_HOME=/nix/store/14lpa5fbiyps2dw9xdrk2l7p8vl1qnhn-temurin-bin-17.0.9  # any temurin-17 in /nix/store
export PATH=$JAVA_HOME/bin:$PATH
export ANDROID_HOME=~/android-sdk
cd android && ./gradlew testDebugUnitTest --console=plain
```

One-time SDK bootstrap (fast, ~1 min): download commandlinetools zip from dl.google.com into `~/android-sdk/cmdline-tools/latest`, accept licenses, `sdkmanager "platforms;android-35" "build-tools;34.0.0" "platform-tools"`, and write `sdk.dir=$HOME/android-sdk` to `android/local.properties` (gitignored).

**Why:** builds otherwise only run in GitHub Actions CI; local runs catch test bugs immediately.

**Gotchas learned the hard way:**
- Kotlin prohibits `vararg` params of type `kotlin.Result<T>` — use `List<Result<T>>`.
- mockk-mocking the final `NovaCycleRepository` whose suspend fun returns `kotlin.Result` misbehaved (wrong answers + OOM). Repository is now `open` with `open suspend fun getHealth()` so tests use a real fake subclass.
- Testing a ViewModel with an infinite `while(isActive) { poll; delay(60s) }` loop under `runTest`: the loop never idles, so runTest cleanup advances virtual time forever (hang). Always cancel `viewModel.viewModelScope` in a `finally` block inside the test.
- Poll-loop timing under virtual time: the first poll fires at t=0 (`runCurrent()`), each `advanceTimeBy(60_001)` afterwards runs exactly one more poll; a naive first `advanceTimeBy` runs TWO polls.
- ShellExec kills background processes when the command exits — long gradle runs must stay in the foreground with `timeout N`.
- Fake-subclassing a repository whose base class has a property initializer touching its DataStore (e.g. `val flow = dataStore.data.map{...}`): the base initializer runs even when the property is overridden, so the mocked DataStore must be `mockk(relaxed = true)` or construction throws "no answer found for getData()".
- Preferences DataStore in JVM tests: to "reopen" a store on the same file you must `job.cancelAndJoin()` the old store's scope first — plain `cancel()` is async, the single-instance file guard stays held, and the new store's reads fail (swallowed by `catch { emptyPreferences() }`, yielding defaults).
