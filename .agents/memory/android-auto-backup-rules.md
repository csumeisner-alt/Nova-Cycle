---
name: Android Auto Backup rules
description: How theme/tap progress survives reinstall; explicit-include semantics of backup rules
---

# Auto Backup for theme progress

The tap achievement / theme unlock prefs (`nova_prefs`) survive reinstall via
Android Auto Backup; Mint Luxe survives via Play `queryPurchasesAsync` restore
regardless of backup.

**Rule:** `backup_rules.xml` (API ≤ 30) and `data_extraction_rules.xml`
(API 31+, both `cloud-backup` and `device-transfer`) use *explicit includes* —
once any `<include>` exists, everything not listed is EXCLUDED. Currently only
`nova_prefs.xml` and the `novacycle_settings` DataStore are backed up.

**Why:** Room cache should be refetched, and FCM registration state must NOT
be restored to a new device (fresh token needed). Adding a new persistent
store users would expect to survive reinstall requires adding it to BOTH xml
files.

**How to apply:** whenever a new SharedPreferences/DataStore file with durable
user progress is added, update both rule files; verify with
`adb shell bmgr backupnow` per `android/REINSTALL_RESTORE_VERIFICATION.md`.
