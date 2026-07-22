"""
NovaCycle Model Trainer
========================
Orchestrates training of both the long-trend (XGBoost) and
short-trend (Keras) models from data stored in the database.

Schedule: retrain weekly (checked on each startup).
"""

import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.models import ModelMetadata, VooCandle, VixCandle
from indicators.technical import TechnicalIndicators
from ml.long_trend import LongTrendModel
from ml.short_trend import ShortTrendModel

logger = logging.getLogger(__name__)

_RETRAIN_INTERVAL_DAYS = 7


class ModelTrainer:
    """Trains and periodically retrains the NovaCycle ML models."""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.long_model = LongTrendModel()
        self.short_model = ShortTrendModel()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def run_initial_training(self, db_session: AsyncSession) -> None:
        """
        Load all historical data from DB, train both models, and log results
        to the ModelMetadata table.

        Steps:
          1. Load daily VOO candles (regular-hours only)
          2. Load VIX candles
          3. Compute all indicators
          4. Train long-trend XGBoost model
          5. Load 5-min VOO candles
          6. Train short-trend Keras model
          7. Persist ModelMetadata rows for both models
        """
        logger.info("Starting initial model training…")
        ticker = settings.TICKER

        # ── Load daily VOO candles ─────────────────────────────────────────────
        daily_df = await self._load_daily_voo(db_session)
        if daily_df.empty:
            logger.warning("No daily VOO data available for training. Skipping.")
            return

        # ── Load VIX candles ───────────────────────────────────────────────────
        vix_df = await self._load_vix(db_session)

        # ── Compute indicators ─────────────────────────────────────────────────
        logger.info("Computing indicators for training data…")
        try:
            indicators = self.indicators.compute_all(
                daily_df, vix_df, exclude_extended=True
            )
        except Exception as exc:
            logger.error("Indicator computation failed: %s", exc)
            indicators = {}

        # ── Train long-trend model ─────────────────────────────────────────────
        logger.info("Training long-trend model…")
        try:
            long_result = self.long_model.train(daily_df, indicators)
            await self._save_metadata(
                db_session,
                model_name="long_trend",
                ticker=ticker,
                accuracy=long_result.get("accuracy", 0.0),
                feature_importances=long_result.get("feature_importances", {}),
            )
            logger.info(
                "Long-trend training complete: accuracy=%.4f",
                long_result.get("accuracy", 0.0),
            )
        except Exception as exc:
            logger.error("Long-trend training failed: %s", exc)

        # ── Load 5-min VOO candles ─────────────────────────────────────────────
        fivemin_df = await self._load_fivemin_voo(db_session)

        if fivemin_df.empty:
            logger.warning("No 5-min VOO data available. Skipping short-trend training.")
            return

        # ── Compute short indicators ───────────────────────────────────────────
        try:
            short_indicators = self.indicators.compute_all(
                fivemin_df, vix_df, exclude_extended=False
            )
        except Exception as exc:
            logger.error("Short indicator computation failed: %s", exc)
            short_indicators = {}

        # ── Train short-trend model ────────────────────────────────────────────
        logger.info("Training short-trend model…")
        try:
            short_result = self.short_model.train(fivemin_df, short_indicators)
            await self._save_metadata(
                db_session,
                model_name="short_trend",
                ticker=ticker,
                accuracy=short_result.get("accuracy", 0.0),
                feature_importances={},
            )
            logger.info(
                "Short-trend training complete: accuracy=%.4f  val_accuracy=%.4f",
                short_result.get("accuracy", 0.0),
                short_result.get("val_accuracy", 0.0),
            )
        except Exception as exc:
            logger.error("Short-trend training failed: %s", exc)

        logger.info("Initial model training complete.")

    async def retrain_if_needed(self, db_session: AsyncSession) -> bool:
        """
        Check whether models were last trained more than RETRAIN_INTERVAL_DAYS ago.
        If so, kick off retraining.

        Returns:
            True if retraining was performed, False otherwise.
        """
        last_trained = await self._get_last_trained(db_session)

        if last_trained is None:
            logger.info("No previous training found. Running initial training.")
            await self.run_initial_training(db_session)
            return True

        now = datetime.utcnow()
        age_days = (now - last_trained).days

        if age_days >= _RETRAIN_INTERVAL_DAYS:
            logger.info(
                "Models last trained %d days ago (threshold=%d). Retraining…",
                age_days,
                _RETRAIN_INTERVAL_DAYS,
            )
            await self.run_initial_training(db_session)
            return True
        else:
            logger.info(
                "Models are up-to-date (last trained %d days ago). No retraining needed.",
                age_days,
            )
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def _load_daily_voo(db_session: AsyncSession) -> pd.DataFrame:
        """Load all daily regular-hours VOO candles from the database."""
        try:
            result = await db_session.execute(
                select(VooCandle).where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == "daily",
                    VooCandle.is_extended_hours == False,
                ).order_by(VooCandle.timestamp.asc())
            )
            rows = result.scalars().all()
            if not rows:
                return pd.DataFrame()

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
            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:
            logger.error("_load_daily_voo error: %s", exc)
            return pd.DataFrame()

    @staticmethod
    async def _load_fivemin_voo(db_session: AsyncSession) -> pd.DataFrame:
        """Load all 5-min VOO candles (including extended hours) from the database."""
        try:
            result = await db_session.execute(
                select(VooCandle).where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == "5min",
                ).order_by(VooCandle.timestamp.asc())
            )
            rows = result.scalars().all()
            if not rows:
                return pd.DataFrame()

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
                    "gap_percent": r.gap_percent or 0.0,
                    "gap_type": r.gap_type,
                }
                for r in rows
            ]
            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:
            logger.error("_load_fivemin_voo error: %s", exc)
            return pd.DataFrame()

    @staticmethod
    async def _load_vix(db_session: AsyncSession) -> pd.DataFrame:
        """Load VIX daily candles from the database."""
        try:
            result = await db_session.execute(
                select(VixCandle).where(
                    VixCandle.ticker == settings.VIX_TICKER,
                    VixCandle.timeframe == "daily",
                ).order_by(VixCandle.timestamp.asc())
            )
            rows = result.scalars().all()
            if not rows:
                return pd.DataFrame()

            records = [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume or 0.0,
                }
                for r in rows
            ]
            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:
            logger.error("_load_vix error: %s", exc)
            return pd.DataFrame()

    @staticmethod
    async def _save_metadata(
        db_session: AsyncSession,
        model_name: str,
        ticker: str,
        accuracy: float,
        feature_importances: dict,
    ) -> None:
        """Persist a ModelMetadata training record."""
        try:
            record = ModelMetadata(
                model_name=model_name,
                ticker=ticker,
                trained_at=datetime.utcnow(),
                accuracy=accuracy,
                feature_importances=json.dumps(feature_importances),
            )
            db_session.add(record)
            await db_session.flush()
        except Exception as exc:
            logger.error("_save_metadata error: %s", exc)

    @staticmethod
    async def _get_last_trained(db_session: AsyncSession):
        """Return the most recent trained_at datetime across all models."""
        try:
            result = await db_session.execute(
                select(func.max(ModelMetadata.trained_at))
            )
            return result.scalar()
        except Exception as exc:
            logger.error("_get_last_trained error: %s", exc)
            return None
