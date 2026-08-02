---
name: Long-model validation and VIX ingestion
description: Honest forward-label validation, model acceptance gating, and VIX-specific OHLC handling
---

Long-model threshold sweeps must construct labels with future prices (`shift(-horizon)`) and align labels by the exact feature index returned after feature construction. A positive result from past-return labels is leakage and must not guide retraining.

**Why:** A standalone sweep initially looked strong only because it used the wrong shift direction. Correct purged walk-forward evaluation showed the 21-day meaningful-move candidates were below the majority baseline.

**How to apply:** Require positive accuracy lift versus the majority baseline before replacing the active long model. Keep the last accepted model when the OOS gate fails, and report the failed attempt as rolled back.

VIX is an index: preserve zero-volume candles, validate intra-bar OHLC consistency, but do not apply the traded-equity cross-bar spike heuristic to VIX. Surface exact VIX/VOO date coverage in health diagnostics.

**Why:** Legitimate VIX regime jumps were quarantined as equity-feed spikes, leaving macro history incomplete even though the vendor supplied the dates.

**How to apply:** Keep the VIX-specific validation exception narrow to historical and targeted VIX ingestion; retain normal spike and volume validation for VOO and SPX futures.

Daily VIX freshness must be measured by lagging VOO trading days, not a fixed calendar-hour threshold. A Friday close can be more than 48 elapsed hours old on Sunday while still being the latest valid market observation.

**Why:** The macro-safety endpoint independently used a 48-hour rule and falsely warned Android users that VIX was stale over the weekend even though health reported zero trading-day lag.

**How to apply:** Keep raw elapsed hours as informational only; use the shared trading-day staleness result for `vix_is_stale` and warning decisions.