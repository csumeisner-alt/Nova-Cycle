---
name: Retrain accuracy independence
description: Relationship between model retrain accuracy history and completed trade-cycle analytics
---

Model retrain accuracy is an offline training metric and must be displayed independently of completed BUY→SELL trade cycles. A new installation can have multiple retrains and accuracy points before it produces its first actionable BUY or SELL cycle.

**Why:** The dashboard previously hid the entire performance content when total completed trades was zero, which also hid valid retrain history and made it appear that no learning data existed.

**How to apply:** Keep trade-specific charts and summaries honest with empty states, but render the long/short accuracy-over-time chart whenever `accuracy_history` contains usable entries.