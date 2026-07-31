"""
Seed realistic BUY→SELL signal history rows into the dev DB so the
Performance tab / Reliability screens can be verified end-to-end with
non-empty data (Task: confirm performance charts with real trades).

Idempotent: deletes previously-seeded rows (cycle_id LIKE 'seed-%') first.
Only inserts SignalHistory rows — trade cycles, P&L, calibration, missed
rallies etc. are all derived by the real engines from these plus the
existing VOO candles already in the DB.

Run:  cd artifacts/api-server/backend && python scripts/seed_performance_demo.py
"""
import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select, and_  # noqa: E402
from database.db import AsyncSessionLocal  # noqa: E402
from database.models import SignalHistory, VooCandle, TradeCycles  # noqa: E402

random.seed(42)


async def main() -> None:
    async with AsyncSessionLocal() as session:
        # Clean previous seeds (and derived persisted cycles) for idempotency
        await session.execute(
            delete(SignalHistory).where(SignalHistory.cycle_id.like("seed-%"))
        )
        await session.execute(delete(TradeCycles))
        await session.commit()

        # Use real 5-min candle timestamps so price lookups resolve properly
        result = await session.execute(
            select(VooCandle.timestamp)
            .where(and_(VooCandle.ticker == "VOO", VooCandle.timeframe == "5min"))
            .order_by(VooCandle.timestamp)
        )
        candle_ts = [r[0] for r in result.all()]
        if len(candle_ts) < 500:
            raise SystemExit("Not enough 5-min candles to seed against.")

        # Spread ~14 cycles over the last ~5 weeks of candles with HOLD gaps
        # between them (gaps allow missed-rally detection to find events).
        recent = [t for t in candle_ts if t >= candle_ts[-1] - timedelta(days=35)]
        rows = []
        n_cycles = 14
        stride = max(1, len(recent) // (n_cycles * 6))
        idx = 0
        confidences = [
            0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65,
            0.72, 0.75, 0.8, 0.85, 0.9, 0.62, 0.78,
        ]
        sessions = ["regular", "regular", "regular", "pre_market", "after_hours"]
        for i in range(n_cycles):
            buy_i = idx + random.randint(1, stride)
            hold = random.randint(3, 24)  # 15 min – 2 hours in 5-min bars
            sell_i = buy_i + hold
            if sell_i >= len(recent) - 2:
                break
            buy_ts, sell_ts = recent[buy_i], recent[sell_i]
            conf = confidences[i % len(confidences)]
            rows.append(SignalHistory(
                timestamp=buy_ts, ticker="VOO", cycle_id=f"seed-{i}",
                signal_type="buy", gauge_type="short", confidence=conf,
                session_type=random.choice(sessions),
                is_extended_hours=False, gap_type="none",
                liquidity_score=round(random.uniform(0.5, 1.0), 2),
                macro_override_applied=False,
            ))
            rows.append(SignalHistory(
                timestamp=sell_ts, ticker="VOO", cycle_id=f"seed-{i}",
                signal_type="sell", gauge_type="short",
                confidence=round(random.uniform(0.4, 0.9), 2),
                session_type="regular", is_extended_hours=False,
                gap_type="none", liquidity_score=round(random.uniform(0.5, 1.0), 2),
                macro_override_applied=False,
            ))
            # leave a HOLD gap before the next buy
            idx = sell_i + random.randint(stride, stride * 4)

        session.add_all(rows)
        await session.commit()
        print(f"Seeded {len(rows)} signal rows ({len(rows)//2} BUY→SELL cycles)")
        print("First buy:", rows[0].timestamp, " Last sell:", rows[-1].timestamp)


if __name__ == "__main__":
    asyncio.run(main())
