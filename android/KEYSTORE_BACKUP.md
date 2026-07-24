# Release Keystore — Backup & Recovery Guide

> **Read this before anything breaks.**  
> If the signing key is ever lost, Android will refuse to install updates over the existing app —
> users would have to uninstall and reinstall from scratch, losing their local data.

---

## What the keystore is

The NovaCycle release APK is signed with a PKCS12 keystore (`.p12` file).  
The keystore lives as a base64-encoded GitHub Actions secret (`KEYSTORE_BASE64`).  
Three companion secrets hold the credentials needed to use it:

| Secret name | What it holds |
|---|---|
| `KEYSTORE_BASE64` | The `.p12` keystore file, base64-encoded |
| `KEYSTORE_PASSWORD` | Password that protects the keystore file itself |
| `KEY_ALIAS` | The name of the signing key entry inside the keystore |
| `KEY_PASSWORD` | Password for that specific key entry |

These are referenced in `.github/workflows/build-apk.yml` and decoded at build time in
`android/app/build.gradle.kts` (the `signingConfigs` block).

---

## Why a backup matters

GitHub Actions secrets can be deleted accidentally, rotated without a copy, or lost when
a repository is transferred or archived.  If `KEYSTORE_BASE64` disappears and no backup
exists, you **cannot** sign future APKs with the same key.  Android enforces that an update
must be signed with the same certificate as the installed version — a different key means
users must uninstall first.

---

## Where to keep the backup

Store the keystore and all four credential values in **at least one** of the following:

| Option | How |
|---|---|
| **Password manager** (recommended) | Bitwarden, 1Password, or similar — create one secure note that holds the base64 blob and the three credential strings. |
| **Encrypted cloud storage** | A file in an encrypted vault (e.g. Cryptomator + Google Drive / Dropbox). Store the `.p12` file and a companion `.txt` with alias and passwords. |
| **Secondary secrets store** | A separate GitHub repo's Actions secrets, or an external vault (e.g. HashiCorp Vault, AWS Secrets Manager). |

> Never store the keystore alongside the source code in a public repo, and never commit it
> unencrypted anywhere.

---

## How to export the keystore from GitHub Actions

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**.
2. GitHub does not let you read an existing secret value through the UI.  
   If you set the secret yourself, you should still have the original base64 string — save that.
3. If you no longer have the original, you can retrieve it during a CI run by adding a
   temporary debug step that prints `$KEYSTORE_BASE64` to the log (mark the run as private /
   delete the step immediately after).

### Decode and verify locally

```bash
# Decode to a file
echo "$KEYSTORE_BASE64" | base64 -d > novacycle-release.p12

# Confirm the key entry is present
keytool -list -v -keystore novacycle-release.p12 -storetype PKCS12 \
        -storepass "$KEYSTORE_PASSWORD"
# You should see: Alias name: <KEY_ALIAS>, Certificate fingerprints, etc.
```

---

## What to do if the keystore is lost

### Scenario A — backup exists

1. Retrieve the base64-encoded keystore from your secure backup location.
2. In GitHub repo → **Settings** → **Secrets and variables** → **Actions**, delete the old
   `KEYSTORE_BASE64` secret and re-add it with the backed-up value.
3. Re-add `KEYSTORE_PASSWORD`, `KEY_ALIAS`, and `KEY_PASSWORD` from the same backup.
4. Trigger the workflow (`workflow_dispatch`) and confirm the APK builds and installs as an
   update over the existing installed version.

### Scenario B — keystore is truly gone (no backup)

You must generate a **new** keystore.  Consequences:
- Existing installs **cannot be updated** silently — users must uninstall and reinstall.
- Any future Play Store listing (if you publish) would require a new app listing or support
  ticket to Google to swap the signing key.

Steps to generate a new keystore:

```bash
keytool -genkey -v \
  -keystore novacycle-release-new.p12 \
  -storetype PKCS12 \
  -alias novacycle \
  -keyalg RSA \
  -keysize 2048 \
  -validity 10000
```

Then base64-encode it and update all four GitHub Actions secrets:

```bash
base64 -w 0 novacycle-release-new.p12   # Linux
# or
base64 -i novacycle-release-new.p12     # macOS
```

Update `KEY_ALIAS` if you chose a different alias during generation.

**After generating a new key, immediately store it in a secure backup location (see above).**

---

## Checklist — do this now if you haven't already

- [ ] Copy `KEYSTORE_BASE64` value to a password manager / encrypted vault
- [ ] Record `KEYSTORE_PASSWORD`, `KEY_ALIAS`, and `KEY_PASSWORD` in the same secure note
- [ ] Verify the backup by decoding and running `keytool -list` on a local machine
- [ ] Confirm at least one team member (other than yourself) also has access to the backup
- [ ] *(Automated)* The `verify-keystore` GitHub Actions workflow runs on the 1st of every month and emails you on failure — no manual calendar reminder needed

---

## Related files

- `.github/workflows/build-apk.yml` — CI workflow that decodes and uses the keystore
- `android/app/build.gradle.kts` — `signingConfigs` block that reads the four env vars
- `android/README_BUILD.md` — general build instructions
