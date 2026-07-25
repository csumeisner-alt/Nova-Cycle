---
name: Prediction router neutral fallbacks
description: Why /api predictions can silently return 0.5 and how model reload works
---

Rules learned fixing the "live backend serves neutral 0.5" incident:

- Prediction routes must pass `build_latest_features()` (a single feature vector) to `predict()`, never the `(X, weights)` tuple from `build_features()`. Passing the tuple raises inside a swallowed try/except and silently degrades to 0.5.
- **Why:** silent except blocks around ML calls masked the tuple bug in production for weeks; always log the exception in those handlers.
- Model classes cache `_model_loaded=True` even when the .pkl file is absent. They now reload based on file mtime (`_maybe_reload`), so a retrain in the same process is picked up without a restart — keep that pattern for any new model class.
- `retrain_if_needed()` must check model files on disk, not just ModelMetadata timestamps: the committed SQLite DB can carry fresh `trained_at` rows while the deployment image lacks the .pkl files.
- `indicators["vix_regime"]` is a per-row pandas Series; the scalar regime is `indicators["latest"]["vix_regime"]`. Passing the Series into string handling (`.upper()`) 500s endpoints.
