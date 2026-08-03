---
name: Rally-event definition unification
description: Shared rally-event label (window-max +0.3% within 12 bars) drives short-model training, walk-forward eval, and missed-rally reporting
---

- One definition lives in `rally_event.py` (backend root): +0.3% rise in max close within the next 12 five-minute bars, strict `>`, NaN for incomplete tail windows. Training labels, walk-forward eval, and missed-rally detection must all import from it — never redefine locally.
- **Why:** the short model previously trained on "close exactly 12 bars later" while the missed-rally detector used window-max, so the model literally wasn't trained to catch the rallies it was blamed for missing (Aug 3 2026 VOO rally incident).
- Evaluate the short model as a rare-event ranker: PR-AUC vs positive base rate, event precision/recall. Majority accuracy/lift-vs-majority is misleading for a ~18–20% positive-rate event; a trainer gate rejects candidates whose OOS PR-AUC ≤ base rate.
- Short `build_features` skips rows (close ≤ 0, per-row exceptions), so it returns `valid_pos`; labels must be indexed `y = labels[valid_pos]` — a positional slice `[:len(X)]` silently misaligns X/y if any mid-series row is skipped.
- Prediction endpoints expose additive `model_state` / `prediction_reliable` fields (healthy | model_unavailable | training_stuck | stale_rolled_back) so stale/stuck models are visibly unreliable; scalar `ml_confidence` must remain for Android compat.
- **How to apply:** any change to rally thresholds/horizon goes only in rally_event.py; any new consumer of the event definition imports it; any feature-builder that can skip rows must return kept positions.
