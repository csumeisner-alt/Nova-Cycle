"""
Generate conviction_fixture.json from live threshold constants.
=================================================================
Run this script whenever you need to regenerate the static fixture file
(e.g. after updating thresholds in signal_engine/conviction.py):

    cd artifacts/api-server/backend
    python tests/fixtures/make_conviction_fixture.py

The generated fixture is used by:
  - scripts/backtest_conviction_tiers.py --fixture tests/fixtures/conviction_fixture.json
  - tests/test_conviction_tiers.py (via the conviction_signals conftest fixture)

Score values are expressed as offsets from AGREEMENT_BAND and other threshold
constants so they stay in sync automatically if the constants change.  If you
change MIN_CYCLE_QUALITY, MIN_ML_CONFIDENCE, or AGREEMENT_BAND in conviction.py,
re-running this script will regenerate a fixture that still exercises the same
logical cases (high-conviction, opportunity-only, neutral, etc.).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly or via tests/ path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from signal_engine.conviction import (
    AGREEMENT_BAND,
    EXIT_MIN_CYCLE_QUALITY,
    EXIT_MIN_ML_CONFIDENCE,
    MIN_CYCLE_QUALITY,
    MIN_ML_CONFIDENCE,
)


def make_signals() -> list[dict]:
    """Return signal dicts with scores derived from live threshold constants.

    Every score that must sit above (or below) a threshold is written as
    ``THRESHOLD ± offset`` so a threshold change propagates automatically.

    Scenarios covered:
      1.  High-conviction BUY  — long gauge, calm, all criteria met
      2.  High-conviction BUY  — long gauge, trending, borderline quality (above entry)
      3.  High-conviction SELL — short gauge, calm, confident sell ML
      4.  Opportunity BUY      — long gauge, calm, low cycle quality (below entry)
      5.  Opportunity BUY      — short gauge, compressed regime (unfavorable)
      6.  Opportunity SELL     — short gauge, macro-shock, marginal scores
      7.  Opportunity BUY      — long gauge, ml_fallback=True (never earns HC)
      8.  Neutral signal        — no tier assigned
      9.  High-return BUY      — long gauge, calm, all criteria met
      10. High-conviction SELL — long gauge, trending
      11. Opportunity BUY      — short gauge, calm, long_score just above EXIT_BAND
      12. Opportunity BUY      — long gauge, calm, quality between exit and entry bands
    """
    # ── Score anchors derived from live constants ──────────────────────────────
    # "Clearly above band" for buy agreement (long + short both point the same way)
    HC_LONG_BUY = round(AGREEMENT_BAND + 62.0, 1)    # e.g. 72 @ band=10
    HC_SHORT_BUY = round(AGREEMENT_BAND + 35.0, 1)   # e.g. 45 @ band=10
    HC2_LONG_BUY = round(AGREEMENT_BAND + 52.0, 1)   # e.g. 62 @ band=10
    HC2_SHORT_BUY = round(AGREEMENT_BAND + 20.0, 1)  # e.g. 30 @ band=10

    # "Clearly below -band" for sell agreement
    HC_LONG_SELL = round(-(AGREEMENT_BAND + 55.0), 1)  # e.g. -65 @ band=10
    HC_SHORT_SELL = round(-(AGREEMENT_BAND + 45.0), 1) # e.g. -55 @ band=10

    # Long gauge just above EXIT band but below AGREEMENT_BAND (no buy agreement)
    WEAK_LONG = round(AGREEMENT_BAND - 5.0, 1)   # e.g. 5 @ band=10

    # Opportunity-zone sell (above -AGREEMENT_BAND, below -EXIT_AGREEMENT_BAND=0)
    OPP_SELL_LONG = round(-(AGREEMENT_BAND + 10.0), 1)  # e.g. -20 @ band=10
    OPP_SELL_SHORT = round(-(AGREEMENT_BAND + 51.0), 1) # e.g. -61 @ band=10

    # Short-gauge BUY: only short_score needs to be above band
    SHORT_GAUGE_SHORT_SCORE = round(AGREEMENT_BAND + 52.0, 1)   # e.g. 62 @ band=10
    SHORT_GAUGE_LONG_SCORE = round(AGREEMENT_BAND + 30.0, 1)    # e.g. 40 @ band=10

    # A second short-gauge BUY just above EXIT_AGREEMENT_BAND (=0) but below band
    ALMOST_BAND_SHORT = round(AGREEMENT_BAND + 53.0, 1)         # e.g. 63 @ band=10
    ALMOST_BAND_LONG = round(AGREEMENT_BAND + 5.0, 1)           # e.g. 15 @ band=10

    # ── Quality and ML anchors ─────────────────────────────────────────────────
    HIGH_QUALITY = round(min(MIN_CYCLE_QUALITY + 0.20, 0.98), 2)  # e.g. 0.85
    MID_QUALITY  = round(MIN_CYCLE_QUALITY + 0.15, 2)             # e.g. 0.80
    ABOVE_EXIT_QUALITY = round(
        (EXIT_MIN_CYCLE_QUALITY + MIN_CYCLE_QUALITY) / 2.0, 2
    )  # between exit and entry bands → stays HC via hysteresis but can't enter
    LOW_QUALITY  = round(EXIT_MIN_CYCLE_QUALITY - 0.10, 2)        # e.g. 0.45

    HIGH_ML_BUY  = round(min(MIN_ML_CONFIDENCE + 0.25, 0.98), 2)  # e.g. 0.90
    HIGH_ML_SELL = round(1.0 - min(MIN_ML_CONFIDENCE + 0.25, 0.98), 2)  # e.g. 0.10
    MID_ML       = round(MIN_ML_CONFIDENCE + 0.10, 2)              # e.g. 0.75 → above entry
    MID_ML_SELL  = round(1.0 - (MIN_ML_CONFIDENCE + 0.10), 2)     # e.g. 0.25
    LOW_ML_BUY   = round(max(MIN_ML_CONFIDENCE - 0.20, 0.05), 2)  # e.g. 0.45 → below entry
    NEUT_ML      = 0.50

    HIGH_ML_BUY2 = round(min(MIN_ML_CONFIDENCE + 0.23, 0.97), 2)  # slight variant
    HIGH_ML_BUY3 = round(min(MIN_ML_CONFIDENCE + 0.21, 0.97), 2)

    return [
        # 1. Strong long-gauge BUY → high conviction
        {
            "signal_type": "buy",  "gauge_type": "long",
            "volatility_regime": "calm",
            "cycle_quality_score": HIGH_QUALITY, "ml_confidence": HIGH_ML_BUY,
            "ml_fallback": False,
            "long_score": HC_LONG_BUY, "short_score": HC_SHORT_BUY,
            "return_percent": 1.8,
        },
        # 2. Long-gauge BUY, trending → high conviction (quality mid-range)
        {
            "signal_type": "buy",  "gauge_type": "long",
            "volatility_regime": "trending",
            "cycle_quality_score": MID_QUALITY, "ml_confidence": HIGH_ML_BUY2,
            "ml_fallback": False,
            "long_score": HC2_LONG_BUY, "short_score": HC2_SHORT_BUY,
            "return_percent": 1.2,
        },
        # 3. Short-gauge SELL → high conviction
        {
            "signal_type": "sell", "gauge_type": "short",
            "volatility_regime": "calm",
            "cycle_quality_score": round(MIN_CYCLE_QUALITY + 0.10, 2),
            "ml_confidence": HIGH_ML_SELL, "ml_fallback": False,
            "long_score": HC_LONG_SELL, "short_score": HC_SHORT_SELL,
            "return_percent": 0.9,
        },
        # 4. Long-gauge BUY, low cycle quality → opportunity only
        {
            "signal_type": "buy",  "gauge_type": "long",
            "volatility_regime": "calm",
            "cycle_quality_score": LOW_QUALITY,
            "ml_confidence": round(MIN_ML_CONFIDENCE - 0.05, 2), "ml_fallback": False,
            "long_score": HC_LONG_BUY - 7.0, "short_score": WEAK_LONG,
            "return_percent": -0.6,
        },
        # 5. Short-gauge BUY, compressed regime → opportunity only
        {
            "signal_type": "buy",  "gauge_type": "short",
            "volatility_regime": "compressed",
            "cycle_quality_score": round(EXIT_MIN_CYCLE_QUALITY + 0.07, 2),
            "ml_confidence": MID_ML, "ml_fallback": False,
            "long_score": SHORT_GAUGE_LONG_SCORE, "short_score": SHORT_GAUGE_SHORT_SCORE,
            "return_percent": -0.3,
        },
        # 6. Short-gauge SELL, macro-shock → opportunity only (marginal agreement)
        {
            "signal_type": "sell", "gauge_type": "short",
            "volatility_regime": "macro_shock",
            "cycle_quality_score": round(EXIT_MIN_CYCLE_QUALITY - 0.05, 2),
            "ml_confidence": MID_ML_SELL, "ml_fallback": False,
            "long_score": OPP_SELL_LONG, "short_score": OPP_SELL_SHORT,
            "return_percent": 0.2,
        },
        # 7. Long-gauge BUY, ml_fallback=True → always opportunity
        {
            "signal_type": "buy",  "gauge_type": "long",
            "volatility_regime": "calm",
            "cycle_quality_score": round(MIN_CYCLE_QUALITY + 0.05, 2),
            "ml_confidence": MID_ML, "ml_fallback": True,
            "long_score": HC_LONG_BUY + 1.0, "short_score": HC2_SHORT_BUY,
            "return_percent": -0.4,
        },
        # 8. Neutral signal → no tier
        {
            "signal_type": "neutral", "gauge_type": "long",
            "volatility_regime": "calm",
            "cycle_quality_score": 0.50, "ml_confidence": NEUT_ML, "ml_fallback": False,
            "long_score": round(AGREEMENT_BAND * 1.0, 1),
            "short_score": round(AGREEMENT_BAND * 0.5, 1),
            "return_percent": None,
        },
        # 9. High-return BUY → high conviction
        {
            "signal_type": "buy",  "gauge_type": "long",
            "volatility_regime": "calm",
            "cycle_quality_score": round(HIGH_QUALITY - 0.03, 2),
            "ml_confidence": round(HIGH_ML_BUY - 0.02, 2), "ml_fallback": False,
            "long_score": round(HC_LONG_BUY + 8.0, 1), "short_score": round(HC_SHORT_BUY + 10.0, 1),
            "return_percent": 2.1,
        },
        # 10. Long-gauge SELL, trending → high conviction
        {
            "signal_type": "sell", "gauge_type": "long",
            "volatility_regime": "trending",
            "cycle_quality_score": round(MIN_CYCLE_QUALITY + 0.13, 2),
            "ml_confidence": HIGH_ML_SELL, "ml_fallback": False,
            "long_score": round(-(AGREEMENT_BAND + 65.0), 1),
            "short_score": round(-(AGREEMENT_BAND + 30.0), 1),
            "return_percent": 1.1,
        },
        # 11. Short-gauge BUY, calm, long_score just above AGREEMENT_BAND
        {
            "signal_type": "buy",  "gauge_type": "short",
            "volatility_regime": "calm",
            "cycle_quality_score": round(EXIT_MIN_CYCLE_QUALITY + 0.03, 2),
            "ml_confidence": round(EXIT_MIN_ML_CONFIDENCE + 0.07, 2), "ml_fallback": False,
            "long_score": ALMOST_BAND_LONG,
            "short_score": ALMOST_BAND_SHORT,
            "return_percent": 0.1,
        },
        # 12. Long-gauge BUY, quality between exit and entry (opportunity only)
        {
            "signal_type": "buy",  "gauge_type": "long",
            "volatility_regime": "calm",
            "cycle_quality_score": ABOVE_EXIT_QUALITY,
            "ml_confidence": round(MIN_ML_CONFIDENCE + 0.01, 2), "ml_fallback": False,
            "long_score": HC2_LONG_BUY, "short_score": round(HC2_SHORT_BUY * 0.1, 1),
            "return_percent": -0.2,
        },
    ]


def main() -> None:
    dest = Path(__file__).parent / "conviction_fixture.json"
    signals = make_signals()
    payload = {"signals": signals}
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(signals)} signals to {dest}")
    # Quick sanity: at least one BUY with long_score above AGREEMENT_BAND
    strong = [
        s for s in signals
        if s["signal_type"] == "buy" and float(s["long_score"]) > AGREEMENT_BAND
    ]
    assert strong, "BUG: no BUY signal above AGREEMENT_BAND after generation"
    print(f"  {len(strong)} BUY signal(s) with long_score > AGREEMENT_BAND ({AGREEMENT_BAND})")
    print(f"  Constants used: AGREEMENT_BAND={AGREEMENT_BAND}, "
          f"MIN_CYCLE_QUALITY={MIN_CYCLE_QUALITY}, "
          f"MIN_ML_CONFIDENCE={MIN_ML_CONFIDENCE}")


if __name__ == "__main__":
    main()
