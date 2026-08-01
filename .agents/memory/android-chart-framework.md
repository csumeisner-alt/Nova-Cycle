---
name: Chart timeframe resampling lessons
description: Non-obvious constraints when serving/rendering multi-timeframe candles across the FastAPI backend and Android app
---

# Multi-timeframe candle lessons

- Resampled bar `volume` must be an integer — the Android model expects a whole number, and floats break parsing only on the resampled path (invisible in daily/5min testing). **Why:** hit exactly this bug during the pro-charting build.
- A time bucket can span two market sessions (regular + after-hours in the 16:00 hour). Emitting groups in session-name order inverts chronology; order resampled bars by each group's first source trade time, never by session label.
- The Android offline candle cache is deliberately daily-only until it becomes timeframe-aware. **Why:** mixed-resolution bars poison the offline fallback. **How to apply:** don't lift the daily-only guard without schema support (follow-up task exists).
- Chart overlays (signals, trade cycles) must map to candles by bucket (last bar start ≤ event time), not exact timestamp equality, or they vanish on coarser timeframes.
