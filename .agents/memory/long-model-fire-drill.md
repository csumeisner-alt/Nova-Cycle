---
name: Long-model promotion fire drill
description: Cross-process retrain tests must validate the complete long-model artifact bundle, not just the pickle.
---

A long-model promotion is only restart-safe when the model pickle, calibrator, calibration report, semantic metadata, and training-status result agree. A rejected candidate must restore or remove every sidecar together with the previous model before a fresh loader is allowed to serve predictions.

**Why:** Testing the pickle and rollback helper independently can miss a stale calibrator, mismatched target metadata, or a failed promotion that leaves the new artifact on disk.

**How to apply:** Keep an isolated fire-drill test around the real trainer orchestration whenever promotion, target semantics, calibration, or rollback behavior changes.