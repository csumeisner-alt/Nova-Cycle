---
name: Short-trend leakage & imbalance lessons
description: Why the 5-min MLP reported 98% accuracy but produced ~0 live probabilities, and rules for honest retraining
---

- Rule: fit feature scalers inside each training window only; never `fit_transform` on the full dataset before a chronological split, and always purge/embargo at least the label horizon (12 bars for the 1h label) between train and test windows.
- Rule: never report/persist train accuracy for a deep net as "accuracy" — an MLP memorizes (~98%) while honest purged walk-forward OOS is far lower. When accuracy semantics change, the trainer's regression-vs-last-success check must skip only the explicit one-time upgrade transition (metric kind is recorded in training status), or the first honest retrain gets rolled back as a false regression.
- Rule: with a rare-event label (>0.3% move in 1h, ~6% base rate), an unbalanced MLP collapses all probabilities to ~1e-6 (pinned −40 gauge term). Oversample the minority class to parity in fold-local fits; healthy result = probs spanning 0–1 with mean ≈ base rate.
- Gotcha: when probing the model from a script, pass real computed indicators (`TechnicalIndicators.compute_all(df, vix_df, exclude_extended=False)`) — empty `indicators={}` produces off-distribution features and fake degenerate probabilities.
**Why:** each of these caused a real failure during the 2026-07 short-trend fix.
**How to apply:** any retrain-pipeline change to either model; reuse the shared walk-forward evaluator and record the accuracy metric kind alongside the accuracy.
