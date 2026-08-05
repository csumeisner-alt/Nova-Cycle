"""
Standalone long-trend retrain script.

Usage (from artifacts/api-server/backend/):
    python scripts/retrain_long_trend.py

Connects directly to the SQLite DB and runs a full long-trend retrain,
then prints the resulting calibration report metadata.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

# Make sure the backend package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("retrain_long_trend")


async def main():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    from config import settings
    from indicators.technical import TechnicalIndicators
    from ml.long_trend import LongTrendModel
    from ml import calibration as ml_calibration

    db_url = settings.DATABASE_URL
    logger.info("Connecting to DB: %s", db_url)

    engine = create_async_engine(db_url, echo=False)
    SessionFactory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionFactory() as session:
        # ── Load daily VOO candles ────────────────────────────────────────────
        from sqlalchemy import select
        from database.models import VooCandle, VixCandle, SpxCandle
        import pandas as pd

        logger.info("Loading daily VOO candles…")
        result = await session.execute(
            select(VooCandle).where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "daily",
                VooCandle.is_extended_hours == False,  # noqa: E712
            ).order_by(VooCandle.timestamp.asc())
        )
        rows = result.scalars().all()
        if not rows:
            logger.error("No daily VOO candles found — aborting.")
            return

        records = [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "is_extended_hours": r.is_extended_hours,
                "session_type": r.session_type,
            }
            for r in rows
        ]
        daily_df = pd.DataFrame(records)
        daily_df.set_index("timestamp", inplace=True)
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df[~daily_df.index.duplicated(keep="last")]
        logger.info("Loaded %d daily VOO candles (%s … %s)",
                    len(daily_df),
                    daily_df.index[0].date(),
                    daily_df.index[-1].date())

        # ── Load VIX candles ─────────────────────────────────────────────────
        logger.info("Loading VIX candles…")
        vix_result = await session.execute(
            select(VixCandle).where(
                VixCandle.ticker == settings.VIX_TICKER,
                VixCandle.timeframe == "daily",
            ).order_by(VixCandle.timestamp.asc())
        )
        vix_rows = vix_result.scalars().all()
        if vix_rows:
            vix_records = [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume or 0.0,
                }
                for r in vix_rows
            ]
            vix_df = pd.DataFrame(vix_records)
            vix_df.set_index("timestamp", inplace=True)
            vix_df.index = pd.to_datetime(vix_df.index)
            vix_df = vix_df[~vix_df.index.duplicated(keep="last")]
            logger.info("Loaded %d VIX candles", len(vix_df))
        else:
            vix_df = pd.DataFrame()
            logger.warning("No VIX candles found — continuing without VIX")

        # ── Load SPX futures close series ─────────────────────────────────────
        spx_close = pd.Series(dtype=float)
        try:
            spx_result = await session.execute(
                select(SpxCandle).where(
                    SpxCandle.ticker == settings.SPX_FUTURES_TICKER,
                    SpxCandle.timeframe == "daily",
                ).order_by(SpxCandle.timestamp.asc())
            )
            spx_rows = spx_result.scalars().all()
            if spx_rows:
                spx_close = pd.Series(
                    [r.close for r in spx_rows],
                    index=pd.to_datetime([r.timestamp for r in spx_rows]),
                    dtype=float,
                )
                spx_close = spx_close[~spx_close.index.duplicated(keep="last")]
                logger.info("Loaded %d SPX futures closes", len(spx_close))
        except Exception as exc:
            logger.warning("SPX futures load skipped: %s", exc)

    # ── Compute indicators ────────────────────────────────────────────────────
    logger.info("Computing technical indicators…")
    indicators_engine = TechnicalIndicators()
    indicators = indicators_engine.compute_all(daily_df, vix_df, exclude_extended=True)
    if not spx_close.empty:
        indicators["spx_futures_close"] = spx_close

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Training long-trend model on %d candles…", len(daily_df))
    model = LongTrendModel()
    result = model.train(daily_df, indicators)

    logger.info("Training result:")
    for k, v in result.items():
        if k not in ("feature_importances", "calibration"):
            logger.info("  %-35s %s", k, v)

    cal = result.get("calibration", {})
    logger.info("Calibration summary:")
    for k, v in cal.items():
        if k not in ("reliability_bins", "folds", "regime_breakdown"):
            logger.info("  %-35s %s", k, v)

    # ── Read back and print the calibration report ────────────────────────────
    report_path = ml_calibration.calibration_report_path("long_trend")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        logger.info("=== Calibration report written to %s ===", report_path)
        logger.info("  stale          : %s", report.get("stale"))
        logger.info("  evaluated      : %s", report.get("evaluated"))
        dataset = report.get("dataset") or {}
        logger.info("  total_candles  : %s", dataset.get("total_candles"))
        logger.info("  labeled_rows   : %s", dataset.get("labeled_rows"))
        logger.info("  date_start     : %s", dataset.get("date_start"))
        logger.info("  date_end       : %s", dataset.get("date_end"))
        logger.info("  oos_accuracy   : %s", report.get("oos_accuracy"))
        logger.info("  positive_rate  : %s", report.get("positive_rate"))
        logger.info("  generated_at   : %s", report.get("generated_at"))
    else:
        logger.error("Calibration report file not found at %s", report_path)

    # ── Post-retrain broader-context ablation (success-gated) ────────────────
    # Only runs when the long-trend retrain produced a valid, non-degenerate
    # model.  Runs the 19- vs 27-feature comparison and appends a timestamped
    # record to ml/models/ablation_broader_context.json.
    # Broader-context series are not yet stored in the DB, so an empty dict
    # is passed; the ablation gracefully falls back to neutral/missing values
    # for the 8 context features — the same behaviour as the standalone script
    # when run without --yf.
    retrain_succeeded = model.model is not None and not result.get("degenerate")
    if not retrain_succeeded:
        logger.warning(
            "Skipping post-retrain ablation: retrain did not produce a valid "
            "model (degenerate=%s, model_is_none=%s).",
            result.get("degenerate"),
            model.model is None,
        )
    else:
        logger.info("Running post-retrain broader-context ablation…")
        try:
            from ml.post_retrain_ablation import run_broader_context_ablation
            ablation_result = run_broader_context_ablation(
                daily_df,
                vix_df,
                spx_close,
                broader_context={},
            )
            if ablation_result:
                passes = ablation_result.get("passes_promotion_gate")
                delta  = ablation_result.get("accuracy_delta_27_minus_19")
                logger.info(
                    "Ablation complete: passes_gate=%s accuracy_delta_27_minus_19=%s",
                    passes, delta,
                )
            else:
                logger.warning("Ablation returned no result (see earlier errors).")
        except Exception as exc:
            logger.error("post_retrain_ablation_error: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
