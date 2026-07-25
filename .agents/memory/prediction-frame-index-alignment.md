---
name: Prediction frame index alignment
description: Router candle DataFrames use integer index + timestamp column; datetime-indexed series must be aligned before feature computation
---

The prediction router loads candle DataFrames with a plain integer index and a `timestamp` column, while the trainer loads them with a DatetimeIndex. Any auxiliary series with a DatetimeIndex (VIX regime, SPX futures close, etc.) passed into feature helpers will fail to `reindex` against the router frames ("Cannot compare dtypes datetime64[ns] and int64") and silently degrade the feature to its default.

**Why:** Feeding the SPX futures series into `compute_macro_sensitivity` worked in training but errored at inference until aligned via the frame's `timestamp` column.

**How to apply:** When injecting a new time series into the prediction path, align it onto the frame's row index first (see `_align_spx_to_df` in the predictions router), and make the feature's consumption block fail soft to its fallback rather than erroring the whole score.
