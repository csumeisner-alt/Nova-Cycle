# Verifying tap achievement & Mint Luxe survive a reinstall

Two independent restore paths cover the theme system:

| Data | Where it lives | Restore mechanism |
|---|---|---|
| Mint Luxe purchase | Google Play (source of truth) | `BillingManager.restorePurchases()` runs `queryPurchasesAsync` on every app start / resume (`MainActivity.onResume`), re-verifies the token with the backend, and flips `mintUnlocked`. Works on any reinstall or new device — no backup needed. |
| 20,000-tap achievement, Aurora/Crimson unlocks, selected theme | SharedPreferences `nova_prefs` | Android Auto Backup. `backup_rules.xml` / `data_extraction_rules.xml` explicitly include `nova_prefs.xml` (and the settings DataStore) for cloud backup **and** device-to-device transfer. |

Note: Auto Backup requires the user to have Google backup enabled on the
device. If a user has backup disabled, tap progress is still lost on
uninstall — Mint Luxe is not, because Play re-delivers it.

## Manual device verification (release checklist)

Auto Backup can be exercised without waiting for the nightly backup, using
the backup manager over adb (debuggable build, backup-enabled device):

```bash
# 1. Generate state: tap the logo a few times, buy/restore Mint if possible.
# 2. Force a backup pass:
adb shell bmgr backupnow com.novacycle

# 3. Full uninstall, then reinstall the same build:
adb uninstall com.novacycle
adb install app-debug.apk

# 4. Launch the app. Expected:
#    - tap count and Aurora/Crimson unlock state restored (from backup)
#    - Mint Luxe re-unlocks within a few seconds (Play restore + server check)

# Inspect restored prefs directly if needed:
adb shell run-as com.novacycle cat shared_prefs/nova_prefs.xml
```

If `bmgr backupnow` reports "Backup is not allowed", enable it:
`adb shell bmgr enable true` and make sure a Google account backup transport
is active (`adb shell bmgr list transports`).
