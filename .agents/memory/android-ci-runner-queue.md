---
name: Android CI runner queue
description: Release validation behavior when GitHub Actions has no available runner
---

GitHub Actions manual Android workflow runs can remain in `queued` indefinitely with no assigned runner even after older queued runs are cancelled.

**Why:** The signed APK workflow is the authoritative release path, but a queued run has not compiled, tested, signed, or published anything.

**How to apply:** Never report a queued Android run as passed and never give the existing rolling APK as the new build. Keep the merged source change in `main`, then rerun the release workflow when a runner is available and verify the release body commit plus APK asset timestamp/hash.