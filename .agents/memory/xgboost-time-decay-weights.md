---
name: XGBoost time-decay weight normalization
description: Raw exp(-λ·age) sample weights over long histories degenerate XGBoost to a constant predictor
---

Time-decay sample weights of the form exp(-λ·age) shrink to ~1e-8 over a decade of daily history. XGBoost's `min_child_weight` compares against the *weighted* hessian sum, so with tiny weights no split ever qualifies and the model trains to a constant prediction — accuracy looks fine (base rate) and all feature importances are exactly 0.

**Why:** The long-trend model reported ~0.70 accuracy but predicted the same probability for every row until weights were normalized to mean 1.0 before fitting.

**How to apply:** Whenever passing recency-decay weights to a gradient-boosted model, normalize them (mean 1.0) first. Red flags of the degenerate case: all-zero feature importances and zero predict_proba variance despite non-trivial accuracy.
