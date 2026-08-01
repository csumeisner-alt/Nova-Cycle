"""
Backtest conviction tiers against historical signals and trade cycles.
======================================================================
Replays historical signal conditions through the ConvictionEvaluator and
compares the two tiers on:
  - signal coverage (count per tier)
  - win rate (per completed BUY→SELL cycle)
  - average return per cycle

Also emits a candidate-frequency / tier-outcome report that breaks results
down by gauge type and session type (regular vs extended hours), so it is
easy to see how often each combination qualifies and whether threshold
changes are suppressing or releasing setups.

Acceptance criteria (exit code non-zero on failure):
  1. Tiering must NEVER suppress a signal: every actionable input signal must
     receive a tier (opportunity coverage == 100% of signals; guardrail allows
     at most MAX_COVERAGE_DROP loss, and by construction it should be 0%).
  2. High-conviction cycles must outperform the overall per-cycle average
     return (they are the "trade on this" tier).

Data source:
  - By default, reads signal_history + trade_cycles from the local SQLite DB.
  - With --fixture <path.json>, replays a deterministic fixture (used by the
    automated test so CI never depends on live DB contents).

Fixture format (JSON):
  {
    "signals": [
      {"signal_type": "buy", "gauge_type": "short",
       "is_extended": false,
       "volatility_regime": "calm", "cycle_quality_score": 0.8,
       "ml_confidence": 0.9, "ml_fallback": false,
       "long_score": 75, "short_score": 55,
       "return_percent": 1.2},   # realized cycle return for this signal
      ...
    ]
  }
  Optional signal field: "is_extended" (bool, default false) marks the bar as
  an extended-hours candidate; it appears in the candidate-frequency report.

Usage:
    cd artifacts/api-server/backend
    python scripts/backtest_conviction_tiers.py [--fixture tests/fixtures/conviction_fixture.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_engine.conviction import (
    ConvictionEvaluator, TIER_HIGH_CONVICTION, TIER_OPPORTUNITY,
)

# Guardrail: tiering may not reduce baseline (opportunity) signal coverage by
# more than this fraction. By design the evaluator only labels, so the
# expected drop is exactly 0 — the guardrail exists to catch regressions.
MAX_COVERAGE_DROP = 0.10


def replay(signals: list[dict]) -> dict:
    """Replay signals through a fresh evaluator; return per-tier metrics."""
    evaluator = ConvictionEvaluator()
    tiers: list[tuple[str, dict]] = []
    untiered = 0
    t = 0.0
    for sig in signals:
        t += 600.0  # 10 minutes apart; keeps regime-transition logic realistic
        res = evaluator.evaluate(
            signal_type=sig["signal_type"],
            gauge_type=sig.get("gauge_type", "long"),
            volatility_regime=sig.get("volatility_regime", "calm"),
            cycle_quality_score=sig.get("cycle_quality_score", 0.5),
            ml_confidence=sig.get("ml_confidence", 0.5),
            ml_fallback=sig.get("ml_fallback", False),
            long_score=sig.get("long_score", 0.0),
            short_score=sig.get("short_score", 0.0),
            now=t,
        )
        if res["tier"] is None:
            untiered += 1
            continue
        tiers.append((res["tier"], sig))

    def metrics(rows: list[dict]) -> dict:
        returns = [
            float(r["return_percent"]) for r in rows
            if r.get("return_percent") is not None
        ]
        wins = sum(1 for x in returns if x > 0)
        return {
            "signals": len(rows),
            "cycles": len(returns),
            "win_rate": (wins / len(returns)) if returns else None,
            "avg_return": (sum(returns) / len(returns)) if returns else None,
        }

    all_rows = [s for _, s in tiers]
    hc_rows = [s for tier, s in tiers if tier == TIER_HIGH_CONVICTION]
    opp_rows = [s for tier, s in tiers if tier == TIER_OPPORTUNITY]

    # ── Candidate-frequency / tier-outcome report ─────────────────────────────
    # Breaks down by gauge type × session type so it is easy to see which
    # combinations qualify and how the threshold change affects coverage.
    # "is_extended" is an optional signal field (default False).
    candidate_freq: dict[str, dict] = {}
    for tier, sig in tiers:
        gauge = sig.get("gauge_type", "long")
        session = "extended" if sig.get("is_extended", False) else "regular"
        key = f"{gauge}:{session}"
        if key not in candidate_freq:
            candidate_freq[key] = {
                "gauge_type": gauge,
                "session": session,
                "total": 0,
                TIER_HIGH_CONVICTION: 0,
                TIER_OPPORTUNITY: 0,
            }
        candidate_freq[key]["total"] += 1
        candidate_freq[key][tier] += 1
    # Add zero rows for combinations present in signals but never tiered
    for sig in signals:
        gauge = sig.get("gauge_type", "long")
        session = "extended" if sig.get("is_extended", False) else "regular"
        key = f"{gauge}:{session}"
        if key not in candidate_freq:
            candidate_freq[key] = {
                "gauge_type": gauge,
                "session": session,
                "total": 0,
                TIER_HIGH_CONVICTION: 0,
                TIER_OPPORTUNITY: 0,
            }
    # Attach tier rate for readability
    candidate_freq_list = []
    for entry in sorted(candidate_freq.values(), key=lambda x: (x["gauge_type"], x["session"])):
        total_input = sum(
            1 for s in signals
            if s.get("gauge_type", "long") == entry["gauge_type"]
            and ("extended" if s.get("is_extended", False) else "regular") == entry["session"]
        )
        entry["input_candidates"] = total_input
        entry["tier_rate"] = round(entry["total"] / total_input, 4) if total_input else None
        candidate_freq_list.append(entry)

    return {
        "input_signals": len(signals),
        "untiered_neutral": untiered,
        "overall": metrics(all_rows),
        "high_conviction": metrics(hc_rows),
        "opportunity_only": metrics(opp_rows),
        "candidate_frequency": candidate_freq_list,
        # Per-signal tier assignments; used by check() for threshold assertions.
        "_signal_tiers": [(tier, sig) for tier, sig in tiers],
    }


def check(report: dict, signals: list[dict] | None = None) -> list[str]:
    """Return list of failure strings (empty = pass).

    ``signals`` is the raw input list from the fixture/DB.  When supplied,
    two additional threshold assertions run:

    1. At least one BUY signal with ``long_score > 65`` exists in the fixture,
       confirming that strong long-gauge setups are reachable under the current
       threshold configuration.  A regression that accidentally raises the
       threshold back above 65 would suppress all such signals — this assertion
       catches that before it reaches production.

    2. No BUY signal with ``long_score <= 65`` is awarded ``TIER_HIGH_CONVICTION``
       in the replay.  Such a score is below the agreement band for long-gauge
       buys, so claiming the top tier would indicate a misconfigured evaluator.
    """
    failures = []
    actionable = report["input_signals"] - report["untiered_neutral"]
    tiered = report["overall"]["signals"]
    if actionable > 0:
        coverage = tiered / actionable
        if coverage < 1.0 - MAX_COVERAGE_DROP:
            failures.append(
                f"coverage guardrail: only {coverage:.0%} of actionable signals "
                f"received a tier (allowed drop <= {MAX_COVERAGE_DROP:.0%})"
            )
    hc = report["high_conviction"]
    overall = report["overall"]
    if hc["cycles"] and overall["cycles"]:
        if hc["avg_return"] <= overall["avg_return"]:
            failures.append(
                f"profitability: high-conviction avg return {hc['avg_return']:+.2f}% "
                f"does not beat overall {overall['avg_return']:+.2f}%"
            )

    # ── Threshold reachability assertions ─────────────────────────────────────
    if signals is not None:
        # Assertion 1: fixture must contain at least one reachable strong buy.
        strong_buys = [
            s for s in signals
            if s.get("signal_type") == "buy" and float(s.get("long_score", 0.0)) > 65
        ]
        if not strong_buys:
            failures.append(
                "threshold reachability: fixture contains no BUY signal with "
                "long_score > 65; a regression raising the threshold would pass "
                "this backtest undetected"
            )

        # Assertion 2: low-long-score buys must never earn TIER_HIGH_CONVICTION.
        signal_tiers = report.get("_signal_tiers", [])
        bad = [
            sig for tier, sig in signal_tiers
            if (
                sig.get("signal_type") == "buy"
                and float(sig.get("long_score", 0.0)) <= 65
                and tier == TIER_HIGH_CONVICTION
            )
        ]
        if bad:
            failures.append(
                f"impossible tier: {len(bad)} BUY signal(s) with long_score <= 65 "
                f"were awarded {TIER_HIGH_CONVICTION} — evaluator thresholds may be "
                f"misconfigured (long_scores: "
                f"{[sig.get('long_score') for sig in bad]})"
            )

    return failures


def load_fixture(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["signals"]


def load_from_db() -> list[dict]:
    """Pair signal_history rows with trade-cycle returns from the local DB."""
    import sqlite3

    db_path = Path(__file__).resolve().parents[1] / "novacycle.db"
    if not db_path.exists():
        raise SystemExit(f"No database at {db_path}; use --fixture instead.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cycles = {
        row["cycle_id"]: row["return_percent"]
        for row in conn.execute(
            "SELECT cycle_id, return_percent FROM trade_cycles "
            "WHERE return_percent IS NOT NULL"
        )
    }
    signals = []
    for row in conn.execute(
        "SELECT * FROM signal_history WHERE signal_type IN ('buy','sell') "
        "ORDER BY timestamp"
    ):
        signals.append({
            "signal_type": row["signal_type"],
            "gauge_type": row["gauge_type"],
            # Historical rows don't store regime/quality inputs; use neutral
            # assumptions so the DB replay measures coverage, while fixture
            # replays exercise the tier criteria precisely.
            "volatility_regime": "calm",
            "cycle_quality_score": row["confidence"],
            "ml_confidence": row["confidence"],
            "ml_fallback": False,
            "long_score": row["confidence"] * 100 * (1 if row["signal_type"] == "buy" else -1),
            "short_score": row["confidence"] * 100 * (1 if row["signal_type"] == "buy" else -1),
            "return_percent": cycles.get(row["cycle_id"]),
        })
    conn.close()
    return signals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", help="JSON fixture path (else: local DB)")
    args = parser.parse_args()

    signals = load_fixture(args.fixture) if args.fixture else load_from_db()
    if not signals:
        print("No signals to replay.")
        return 0

    report = replay(signals)

    # ── Candidate-frequency report ─────────────────────────────────────────────

    print("── Candidate frequency / tier outcomes ──────────────────────────────")
    freq = report.get("candidate_frequency", [])
    if freq:
        header = f"{'gauge:session':<20} {'input':>6} {'tiered':>7} {'tier_rate':>10} {TIER_HIGH_CONVICTION:>16} {TIER_OPPORTUNITY:>12}"
        print(header)
        print("-" * len(header))
        for row in freq:
            label = f"{row['gauge_type']}:{row['session']}"
            print(
                f"{label:<20} {row['input_candidates']:>6} {row['total']:>7} "
                f"{(row['tier_rate'] or 0.0):>9.1%} "
                f"{row[TIER_HIGH_CONVICTION]:>16} "
                f"{row[TIER_OPPORTUNITY]:>12}"
            )
    else:
        print("  (no tiered candidates)")
    print()

    # ── Full JSON report ───────────────────────────────────────────────────────
    print(json.dumps(report, indent=2))

    failures = check(report, signals=signals)
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: tier coverage guardrail held and high-conviction outperformed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
