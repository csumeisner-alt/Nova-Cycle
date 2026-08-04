---
name: Long-trend no directional edge
description: 128-config dry-run grid found no feature/horizon/threshold combo beats the majority baseline for VOO long-trend
---
Rule: at the 21d/±2% target (and 5/10/42d, ±1–3% variants), no feature set or model (XGBoost or logistic, 128 configs) beat the majority "always up" baseline OOS; best lift was −3.7pp (H=5d).
**Why:** VOO 2016–2026 is a persistent bull trend (majority baseline 57–79%); models learn trend patterns that invert OOS. Full evidence in artifacts/api-server/backend/docs/long_trend_feature_exploration.md; harness scripts/long_trend_dry_run.py (read-only DB, /tmp model dir).
**How to apply:** don't retry feature tweaks in this space expecting a gate pass; the recommended path is a calibrated majority-class baseline (~0.73 base rate) as the long signal until fundamentally new data sources exist. Never weaken the OOS gate.

## Financial benchmark rule

Strategy comparisons must retain every out-of-sample decision day. Use future
returns only to score meaningful classification labels; never filter the days
that receive a portfolio position based on their future return. A close-of-day
decision must be applied to the next trading day's return, with purged,
chronological folds for model training.

**Why:** Filtering the benchmark by the future label can make a strategy look
better by silently removing its hardest decisions. The causal benchmark showed
the current model returned +41.15% versus +87.63% for buy-and-hold while using
18.4x annual turnover, so the model is not a financial improvement despite a
modest drawdown reduction.

**How to apply:** Require future-only next-day portfolio simulation and compare
total return, CAGR, Sharpe, maximum drawdown, turnover, and downside capture
against buy-and-hold and simple risk-managed baselines before promotion.
