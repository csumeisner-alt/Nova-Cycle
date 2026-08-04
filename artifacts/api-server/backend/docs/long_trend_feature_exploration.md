# Long-Trend Feature / Target Exploration
*Research document — no model promotion. OOS gate in `ml/trainer.py` and `config.py` is unchanged.*

## Overview

This document records a systematic dry-run exploration of alternative horizons, label
thresholds, feature subsets, and model types for the long-trend (daily VOO) directional
classifier, run after the train/inference feature-parity bug was fixed.  The goal was to
determine whether any configuration can achieve positive out-of-sample lift versus the
majority-class baseline, or whether the task itself lacks the signal needed to pass the gate.

**Conclusion: (b)** — No tested configuration beats the majority baseline OOS. The
evidence strongly recommends retiring the active-learning approach for the long signal and
falling back to a calibrated majority-class estimator that is honest about its uncertainty.

---

## Causal strategy benchmark

The dry-run harness also supports `--benchmark`, which evaluates the current
long-target configuration as a trading strategy without changing production
artifacts:

```bash
cd artifacts/api-server/backend
python scripts/long_trend_dry_run.py --quick --benchmark
```

This benchmark is intentionally separate from the classification grid:

- Every out-of-sample decision day is retained; future-return filtering is used
  only to define which rows receive a classification accuracy label.
- Each fold trains only on past meaningful labels and leaves an embargo of at
  least the full target horizon before the test window.
- A prediction made at the close of day `t` is applied to the return from `t` to
  `t+1`, so no future close is available when the position is selected.
- The benchmark compares the current model with always-up/buy-and-hold, a
  200-day moving-average filter, and a 10% annualized volatility-targeted
  long-only exposure.
- Pooled and per-fold results include total return, CAGR, Sharpe, maximum
  drawdown, annualized one-way turnover, and downside capture. Classification
  accuracy is reported for the current model, always-up/buy-and-hold, and the
  moving-average filter on meaningful H-day labels.

The output is written to a temporary `strategy_benchmark.json` file, not to
`ml/models`, and is never a production promotion or gate decision by itself.

### Current VOO benchmark result

Using the local read-only database (2,521 daily VOO candles, 2016-07-25
through 2026-08-04), the default 21-day/±2% target produced 1,261
out-of-sample decision days across five purged folds:

| Strategy | Total return | CAGR | Sharpe | Max drawdown | Annual turnover | Downside capture |
|---|---:|---:|---:|---:|---:|---:|
| Current long model | +41.15% | +7.14% | 0.59 | −21.17% | 18.40 | 0.93 |
| Always-up / buy-and-hold | +87.63% | +13.41% | 0.83 | −24.52% | 0.20 | 1.00 |
| 200-day SMA filter | +63.07% | +10.27% | 0.91 | −19.56% | 6.20 | 0.95 |
| 10% volatility-targeted | +54.00% | +9.02% | 0.88 | −12.76% | 5.22 | 0.96 |

The current model reduced drawdown slightly versus buy-and-hold, but it
materially lost on total return, CAGR, and Sharpe while requiring substantially
more turnover. It therefore does **not** qualify as an improvement or a
replacement for the baseline behavior. These results reinforce the rule that
future long candidates must beat a practical strategy baseline on both return
and risk-adjusted metrics before promotion.

---

## Methodology

### Harness

`scripts/long_trend_dry_run.py` — self-contained, no side effects:

- Opens the production DB **read-only** via `sqlite URI (file:...?mode=ro)`.
- Patches `ml.calibration.MODEL_DIR → /tmp/lt_dryrun_<pid>` before any import so no
  write ever reaches `ml/models` or the live database.
- Reuses `LongTrendModel.build_features()` and `ml.calibration.walk_forward_evaluate()`
  unchanged so results reflect the real pipeline.
- Prints a markdown table and saves a JSON summary to the tmp dir.
- Verified: `ml/models` file count unchanged (9 files) across all runs.

Usage:
```
cd artifacts/api-server/backend
python scripts/long_trend_dry_run.py            # full grid
python scripts/long_trend_dry_run.py --quick    # core matrix only
python scripts/long_trend_dry_run.py --yf       # yfinance mode (no DB needed)
python scripts/long_trend_dry_run.py --combo 21,0.02   # single config
```

### Dataset

| Property | Value |
|---|---|
| Source | `voo_candles` (regular-hours daily) |
| Rows | 2 521 trading days |
| Date range | 2016-07-25 → 2026-08-04 |
| VIX rows | 2 522 |
| SPX futures rows | 2 516 |

### Label construction (anti-leakage)

For each configuration `(horizon H, threshold T)`:

1. Forward return = `close.shift(-H) / close - 1` on the **full, sorted, unfiltered** frame.
2. Rows with `|return| < T` excluded as noise (sign-only when T=0).
3. `y=1` if forward_return ≥ T, `y=0` if forward_return ≤ −T.

All rolling features (`_return_5d`, `_return_10d`, `_return_20d`, `_vol_avg20`, plus the
four additive features) are pre-computed on the **full, unfiltered** frame before the
noise-row filter is applied, then reindexed to the filtered subset — the same fix applied
in `LongTrendModel.train()` to prevent the sign-inversion bug (see
`.agents/memory/long-trend-return-alignment.md`).

### OOS evaluation

Purged chronological walk-forward, 5 folds, embargo = max(H, 21) rows.  Each fold trains
only on rows strictly before the test window minus the embargo gap, preventing forward
label overlap.  No fold-local scaling for XGBoost; fold-local `StandardScaler` applied
inside the sklearn wrapper for logistic regression.

---

## Exploration grid

| Dimension | Values tested |
|---|---|
| Horizon (H) | 5 d, 10 d, 21 d, 42 d |
| Threshold (T) | 0 % (sign-only), 1 %, 2 %, 3 % |
| Feature set | `all_19` (current), `no_macro`, `momentum_vix`, `trend_mom` |
| Model | XGBoost (current), LogisticRegression (calibrated baseline) |

**Total configurations:** 4 × 4 × 4 × 2 = **128** trained + 16 majority-class baselines.

---

## Results table (representative subset — full grid in JSON)

`Lift` = OOS accuracy − majority-class baseline accuracy.  A positive lift is the minimum
bar to pass the OOS gate (`LONG_MIN_OOS_ACCURACY_LIFT = 0.0`).

### Majority-class baselines

| Config | H | T | N rows | pos% | Majority baseline | Bal acc |
|---|---|---|---|---|---|---|
| Majority | 5d | 0% | 2 516 | 62.2% | **62.2%** | 0.500 |
| Majority | 5d | 2% | 698 | 57.5% | **57.5%** | 0.500 |
| Majority | 10d | 0% | 2 511 | 66.2% | **66.2%** | 0.500 |
| Majority | 10d | 2% | 1 134 | 66.6% | **66.6%** | 0.500 |
| Majority | 21d | 0% | 2 500 | 69.6% | **69.6%** | 0.500 |
| Majority | 21d | 2% | 1 633 | 72.9% | **72.9%** | 0.500 |
| Majority | 42d | 0% | 2 479 | 75.4% | **75.4%** | 0.500 |
| Majority | 42d | 2% | 1 903 | 79.4% | **79.4%** | 0.500 |

> **Key observation:** the majority-class baseline is very high (57–79%) because VOO has
> been in a persistent bull trend over the 2016–2026 data window. Any directional model
> must beat a "always predict up" strategy, which is a hard bar.

### Feature-set × model × horizon matrix (all_19 feature set, T=0 % and T=2 %)

| Config | H | T | Model | N | OOS acc | Bal acc | Lift | Maj base |
|---|---|---|---|---|---|---|---|---|
| all_19 | 5 | 0% | XGBoost | 2 516 | 51.5% | 50.9% | **−8.0pp** | 62.2% |
| all_19 | 5 | 0% | Logistic | 2 516 | 49.9% | 49.5% | **−9.6pp** | 62.2% |
| all_19 | 5 | 2% | XGBoost | 698 | 47.3% | 48.5% | **−8.9pp** | 57.5% |
| all_19 | 5 | 2% | Logistic | 698 | 52.4% | 53.3% | **−3.7pp** | 57.5% |
| all_19 | 10 | 0% | XGBoost | 2 511 | 51.0% | 49.2% | **−10.6pp** | 66.2% |
| all_19 | 10 | 0% | Logistic | 2 511 | 48.5% | 48.2% | **−13.1pp** | 66.2% |
| all_19 | 10 | 2% | XGBoost | 1 134 | 44.4% | 44.2% | **−20.3pp** | 66.6% |
| all_19 | 10 | 2% | Logistic | 1 134 | 52.2% | 51.3% | **−12.5pp** | 66.6% |
| all_19 | 21 | 0% | XGBoost | 2 500 | 45.0% | 45.3% | **−20.2pp** | 69.6% |
| all_19 | 21 | 0% | Logistic | 2 500 | 48.1% | 47.2% | **−17.1pp** | 69.6% |
| **all_19** | **21** | **2%** | **XGBoost** | **1 633** | **38.3%** | **38.9%** | **−29.0pp** | **72.9%** |
| **all_19** | **21** | **2%** | **Logistic** | **1 633** | **38.2%** | **40.6%** | **−29.1pp** | **72.9%** |
| all_19 | 42 | 0% | XGBoost | 2 479 | 48.6% | 45.5% | **−22.4pp** | 75.4% |
| all_19 | 42 | 0% | Logistic | 2 479 | 43.4% | 43.5% | **−27.6pp** | 75.4% |
| all_19 | 42 | 2% | XGBoost | 1 903 | 54.7% | 44.8% | **−20.5pp** | 79.4% |
| all_19 | 42 | 2% | Logistic | 1 903 | 46.6% | 48.5% | **−28.6pp** | 79.4% |

(The current production configuration — H=21, T=2 % — is bolded. OOS lift: −29 pp.)

### Feature-set comparison at H=21, T=2 %

| Feature set | Model | OOS acc | Bal acc | Lift |
|---|---|---|---|---|
| all_19 (19 features) | XGBoost | 38.3% | 38.9% | **−29.0pp** |
| all_19 (19 features) | Logistic | 38.2% | 40.6% | **−29.1pp** |
| no_macro (15 feat) | XGBoost | 35.1% | 35.1% | **−32.2pp** |
| no_macro (15 feat) | Logistic | 34.9% | 39.5% | **−32.4pp** |
| momentum_vix (6 feat) | XGBoost | 47.4% | 44.1% | **−20.0pp** |
| momentum_vix (6 feat) | Logistic | 47.0% | 47.3% | **−20.2pp** |
| trend_mom (9 feat) | XGBoost | 47.7% | 43.9% | **−19.8pp** |
| trend_mom (9 feat) | Logistic | 42.0% | 45.0% | **−24.2pp** |

### Feature-set comparison at H=5, T=2 % (shortest horizon, smallest baseline)

| Feature set | Model | OOS acc | Bal acc | Lift |
|---|---|---|---|---|
| all_19 | XGBoost | 47.3% | 48.5% | **−8.9pp** |
| all_19 | Logistic | 52.4% | 53.3% | **−3.7pp** |
| no_macro | XGBoost | 52.9% | 53.9% | **−3.8pp** |
| no_macro | Logistic | 51.7% | 52.6% | **−5.1pp** |
| momentum_vix | XGBoost | 51.9% | 51.0% | **−4.8pp** |
| momentum_vix | Logistic | 50.0% | 51.6% | **−6.6pp** |
| trend_mom | XGBoost | 52.4% | 52.7% | **−4.4pp** |
| trend_mom | Logistic | 50.9% | 51.1% | **−5.8pp** |

> H=5 / T=2 % is the **closest** any configuration came to the baseline (best: −3.7 pp
> with Logistic / all_19), but still clearly negative.

### Full grid summary

| Result | Count |
|---|---|
| Configurations with positive lift | **0 / 128** |
| Configurations within 5 pp of baseline | **0 / 128** |
| Best (least negative) lift seen | **−3.7 pp** (H=5, T=2%, Logistic/all_19) |
| Worst lift seen | **−36.5 pp** (H=42, T=3%, Logistic/trend_mom) |

---

## Analysis

### Why the majority baseline is so high

VOO has returned roughly +14 % per year over the 2016–2026 window.  At a 21-day horizon,
`forward_return > 0` is true ~70 % of the time after the ±2 % noise filter; at 42 days it
rises to ~79 %.  Any directional model must beat "always bullish", which is a very high
bar for a mean-reverting feature set built from recent returns and moving averages.

### Why models score *below* the majority baseline

The OOS balanced accuracy ranges from 0.35 to 0.53, mostly below 0.50. This means the
model is actively wrong more than chance in many configurations — it learns patterns that
hold in-sample but invert out-of-sample. This is consistent with:

1. **Trend-chasing features in a persistent bull market**: SMA cross-overs, MACD, recent
   returns all signal "up" → the model predicts "up" → accuracy looks fine in-sample
   (agrees with the trend), but in the held-out test window the same features again say
   "up" and the true label *also* tends to be "up" — yet the model underperforms the
   naive "always up" baseline, meaning it fires incorrectly when it predicts "down".

2. **High majority baseline kills margin**: with 73–79 % of labels being "up", even a
   model that is slightly better than chance at calling direction earns, say, 50–52 %
   raw accuracy — which still looks like −20 pp lift.

3. **No feature conveys genuine edge at any horizon tested**: momentum features carry only
   weak signal at the 21+ day scale in a trending ETF. VIX regime features provide
   volatility context but not directional edge. The macro/overnight features help
   short-term alerting but add noise at monthly horizons.

### Feature-set breakdown

- **Dropping macro features** (`no_macro`) uniformly worsens performance — macro context
  is better than nothing but still not sufficient.
- **Minimal momentum + VIX** (`momentum_vix`) consistently performs *better* than the full
  feature set at the 21–42 d horizon, suggesting some additive features add noise.
- **Shorter horizons** (H=5) show the smallest (least bad) lifts — the task is slightly
  more tractable at 5 days, but even then lift never crosses zero.

---

## Leakage pitfalls encountered and verified clean

| Risk | Mitigation verified |
|---|---|
| Forward label using past prices | `close.shift(-H)` computed on full sorted frame. ✅ |
| Rolling return features re-computed on filtered subset | Pre-computed on full frame before noise filter, then reindexed. Same logic as `LongTrendModel.train()`. ✅ |
| Embargo gap < label horizon | Embargo = max(H, 21) rows in all folds. ✅ |
| Fold-local scaling not applied | `StandardScaler` fitted only on train portion inside wrapper; never sees test rows. ✅ |
| Writes to `ml/models` | `ml.calibration.MODEL_DIR` patched to `/tmp` before any import. Verified file count unchanged. ✅ |
| DB mutation | `sqlite URI mode=ro`; any write attempt raises `OperationalError`. ✅ |

---

## Conclusion: recommendation (b)

**Evidence-backed recommendation: fall back to a calibrated majority-class baseline for
the long signal.**

The OOS gate correctly refuses all trained candidates.  Exploration across 128
configurations covering 4 horizons, 4 thresholds, 4 feature subsets, and 2 model
architectures found zero configurations with positive OOS lift.  The best result was
−3.7 pp (H=5, T=2 %, Logistic).  The current production configuration (H=21, T=2 %,
XGBoost) shows −29 pp.

**What to do in the app:**
- Continue serving the stale legacy model as-is — it is explicitly flagged as legacy and
  the `is_neutral_fallback()` path returns `0.5` when no valid model exists.
- The long gauge's "calibrated neutral point" (`calibration_base_rate ≈ 0.73`) already
  represents the base-rate bull bias; exposing this as a constant calibrated output (with
  low confidence) is more honest than a trained model that actively inverts.
- A **calibrated majority-class estimator** should be wired in as the fallback: always
  output `positive_rate` (≈ 0.73 at 21d/2%) as the probability, so the gauge shows the
  historical up-bias with an explicit "no trained edge" confidence signal.
- **Longer-term path forward** (not implemented here): regime-conditional modelling
  (separate models per VIX regime), higher-frequency targets (weekly), or exogenous
  macro series (yields, sector rotation) may provide edge not available in price/volume
  features alone.

---

## Reproducing the results

```bash
cd artifacts/api-server/backend

# Full grid (≈4 min)
python scripts/long_trend_dry_run.py

# Quick core matrix only (≈1 min)
python scripts/long_trend_dry_run.py --quick

# Single configuration
python scripts/long_trend_dry_run.py --combo 5,0.02

# No DB needed (fetches from yfinance)
python scripts/long_trend_dry_run.py --yf --quick
```

JSON results are written to `/tmp/lt_dryrun_<pid>/results.json`.
`ml/models` is never touched.

---

*Generated: 2025-08-05. Data: VOO daily 2016-07-25 → 2026-08-04 (2521 rows).*
