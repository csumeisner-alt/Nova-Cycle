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
