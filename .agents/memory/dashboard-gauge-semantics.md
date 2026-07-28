---
name: Dashboard gauge number semantics
description: Two different percentages on the Android gauges — directional position vs model confidence
---

# Gauge number semantics

The Android dashboard gauge shows TWO distinct 0–100% values that must never be conflated in UI copy:

- **Big number** = directional gauge position (score −100..100 mapped to 0..100; 0 = strong sell, 50 = neutral, 100 = strong buy). BUY/HOLD/SELL word derives from it (≥65 BUY, ≤35 SELL).
- **Small "% confidence" line** = model conviction, with zones 0–30 Weak / 31–64 Uncertain / 65–100 Strong.

**Why:** a completion review rejected sheet copy that described the directional number as "confidence" — concurrent work had changed which value the gauge headline shows.

**How to apply:** any explainer text, badge, or web-dashboard copy touching gauges must state which of the two values it describes; re-check the widget's current headline value before writing copy, since concurrent tasks rewrite the gauge widget frequently.
