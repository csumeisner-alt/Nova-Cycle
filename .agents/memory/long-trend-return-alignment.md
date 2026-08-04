---
name: Long-trend return alignment bug
description: train/inference mismatch in return_5d/10d/20d and vol_avg20 features caused sign-inverted OOS predictions and -0.30 lift
---

# Long-trend return alignment bug

The root cause of repeated long-model OOS quality gate failures (-0.3051 lift).

## The Rule
Pre-compute `_return_5d`, `_return_10d`, `_return_20d`, and `_vol_avg20` on the **full, unfiltered** daily df in `LongTrendModel.train()` **before** the meaningful-move filter is applied. `build_features()` consumes these columns (training path) and falls back to `iloc`-based computation only when they are absent (inference path, where the full df is always passed).

## Why
`build_features()` computes returns via `df["close"].iloc[i-5]` — integer position offsets on whatever df is passed. At training time, `train()` passes the meaningful-move-filtered subset (non-contiguous after removing noise rows). So `iloc[i-5]` spans 5 *filtered rows* instead of 5 *trading days*, which can produce a sign-inverted return. At inference time (`build_latest_features`), the full unfiltered df is passed, so `iloc[i-5]` = 5 real trading days. This train/inference mismatch taught the model the wrong direction, causing the model to be anti-correlated with the target.

## How to Apply
- `train()` in `ml/long_trend.py`: add `_return_5d/10d/20d` via `pct_change()` and `_vol_avg20` via `rolling(20).mean()` on the full df right after the regular-hours filter, before the `future_close` shift and `dropna`.
- `build_features()`: check `if "_return_5d" in df.columns:` — use the column value; else use the old `iloc` fallback.
- Tests in `tests/test_long_trend_return_alignment.py` cover: value matches, sign consistency, OOS lift > -0.10, rollback still fires.

**Why the threshold is -0.10, not 0.0**: Random Brownian motion data produces XGBoost variance of ±0.05 even with a correctly aligned model. The real failure mode produces -0.20 to -0.30. -0.10 catches the bug while tolerating noise.

## Extension (Aug 2026): the same bug class hit the additive features
The four additive rolling-window features (`volatility_regime_enc`, `macro_sensitivity_score`, `macro_override_flag`, `overnight_return_weighted`) were still recomputed inside `build_features()` on the filtered subset, so their windows spanned filtered rows. Fix mirrors the return-feature pattern: `train()` pre-computes `_vol_regime_enc`, `_macro_sens`, `_macro_flag`, `_overnight_w` on the full df; `build_features()` consumes them when present. Parity test: `tests/test_long_trend_additive_feature_parity.py` asserts full feature-vector equality between train path and inference path for shared timestamps.

**Lesson**: ANY feature computed inside `build_features()` from the passed df is at risk. When adding a new rolling/window feature, either pre-compute it in `train()` before the meaningful-move filter, or extend the parity test — it will catch the mismatch automatically.

**Important honest result**: after fixing additive-feature parity, isolated dry-run OOS accuracy improved 35.6% → 44.7% but lift vs majority is still -22.6pp (balanced acc 0.417). Feature parity was necessary but not sufficient — with parity clean, the remaining negative lift means the current feature set has no real edge at the 21-day/±2% target; the OOS gate is correctly refusing to promote. Don't weaken the gate; the model needs better features or a different target, not more retries.
