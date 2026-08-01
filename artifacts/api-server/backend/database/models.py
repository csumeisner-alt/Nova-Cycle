"""
NovaCycle ORM Models
====================
All tables include `ticker TEXT NOT NULL DEFAULT 'VOO'` as a placeholder
for future multi-ticker support.
"""

from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# VOO Candlestick data
# Timeframe: 'daily' or '5min'
# Session type: 'pre_market', 'regular', 'after_hours'
# ─────────────────────────────────────────────────────────────────────────────
class VooCandle(Base):
    __tablename__ = "voo_candles"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Ticker placeholder (always 'VOO' for now)
    ticker = Column(String(16), nullable=False, default="VOO", index=True)

    # OHLCV
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)

    # Session metadata
    timeframe = Column(String(8), nullable=False, default="daily")   # 'daily' | '5min'
    is_extended_hours = Column(Boolean, nullable=False, default=False)
    session_type = Column(String(16), nullable=False, default="regular")
    # session_type values: 'pre_market', 'regular', 'after_hours'

    # Gap analysis
    # GapPercent = (PreMarketOpen - PreviousClose) / PreviousClose * 100
    gap_percent = Column(Float, nullable=True)
    gap_type = Column(String(16), nullable=False, default="none")
    # gap_type values: 'gap_up', 'gap_down', 'none'


# ─────────────────────────────────────────────────────────────────────────────
# VIX Candlestick data
# ─────────────────────────────────────────────────────────────────────────────
class VixCandle(Base):
    __tablename__ = "vix_candles"

    id = Column(Integer, primary_key=True, autoincrement=True)

    ticker = Column(String(16), nullable=False, default="^VIX", index=True)

    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True, default=0.0)

    timeframe = Column(String(8), nullable=False, default="daily")


# ─────────────────────────────────────────────────────────────────────────────
# SPX futures Candlestick data (E-mini S&P 500, ES=F)
# ─────────────────────────────────────────────────────────────────────────────
class SpxCandle(Base):
    __tablename__ = "spx_candles"

    id = Column(Integer, primary_key=True, autoincrement=True)

    ticker = Column(String(16), nullable=False, default="ES=F", index=True)

    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True, default=0.0)

    timeframe = Column(String(8), nullable=False, default="daily")


# ─────────────────────────────────────────────────────────────────────────────
# Confidence History
# Stores rolling ML + indicator confidence snapshots
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceHistory(Base):
    __tablename__ = "confidence_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    ticker = Column(String(16), nullable=False, default="VOO", index=True)

    long_buy_confidence = Column(Float, nullable=True)
    long_sell_confidence = Column(Float, nullable=True)
    short_buy_confidence = Column(Float, nullable=True)
    short_sell_confidence = Column(Float, nullable=True)

    session_type = Column(String(16), nullable=False, default="regular")
    is_extended_hours = Column(Boolean, nullable=False, default=False)


# ─────────────────────────────────────────────────────────────────────────────
# Signal History
# Every generated signal (pre-filter)
# ─────────────────────────────────────────────────────────────────────────────
class SignalHistory(Base):
    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    ticker = Column(String(16), nullable=False, default="VOO", index=True)

    cycle_id = Column(String(64), nullable=True)
    signal_type = Column(String(8), nullable=False)    # 'buy' | 'sell'
    gauge_type = Column(String(8), nullable=False)     # 'long' | 'short'
    confidence = Column(Float, nullable=False)

    session_type = Column(String(16), nullable=False, default="regular")
    is_extended_hours = Column(Boolean, nullable=False, default=False)
    gap_type = Column(String(16), nullable=False, default="none")
    liquidity_score = Column(Float, nullable=True)
    macro_override_applied = Column(Boolean, nullable=False, default=False)

    # ── Conviction tier ('opportunity' | 'high_conviction'; NULL for rows
    #    recorded before tiering existed) ─────────────────────────────────────
    conviction_tier = Column(String(24), nullable=True)
    # JSON-encoded list of plain-language reason strings
    conviction_reasons = Column(Text, nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# Trade Cycles
# Pairs of BUY → SELL signals with P&L
# ─────────────────────────────────────────────────────────────────────────────
class TradeCycles(Base):
    __tablename__ = "trade_cycles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(String(64), nullable=False, unique=True, index=True)
    ticker = Column(String(16), nullable=False, default="VOO", index=True)

    buy_timestamp = Column(DateTime, nullable=True)
    sell_timestamp = Column(DateTime, nullable=True)

    buy_price = Column(Float, nullable=True)
    sell_price = Column(Float, nullable=True)

    # return_percent = (sell_price - buy_price) / buy_price * 100
    return_percent = Column(Float, nullable=True)
    # return_dollars = sell_price - buy_price (per share)
    return_dollars = Column(Float, nullable=True)

    hold_time_minutes = Column(Float, nullable=True)

    # Captured from the BUY signal that opened this cycle
    confidence_at_buy = Column(Float, nullable=True)
    confidence_at_sell = Column(Float, nullable=True)
    session_type_at_buy = Column(String(16), nullable=True)   # 'pre_market', 'regular', 'after_hours'
    liquidity_score_at_buy = Column(Float, nullable=True)
    gap_type_at_buy = Column(String(16), nullable=True)
    macro_override_applied = Column(Boolean, nullable=False, default=False)

    # Classifications used for segmented reliability metrics
    volatility_class = Column(String(16), nullable=True)   # e.g. 'low'|'medium'|'high'
    liquidity_class = Column(String(16), nullable=True)    # e.g. 'adequate'|'thin'


# ─────────────────────────────────────────────────────────────────────────────
# Filtered Signal History
# Post-filter: strongest-confidence, alternating BUY/SELL, cycle-tagged
# ─────────────────────────────────────────────────────────────────────────────
class FilteredSignal(Base):
    __tablename__ = "filtered_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    ticker = Column(String(16), nullable=False, default="VOO", index=True)

    signal_type = Column(String(8), nullable=False)    # 'buy' | 'sell'
    gauge_type = Column(String(8), nullable=False)     # 'long' | 'short'
    confidence = Column(Float, nullable=False)
    cycle_id = Column(String(64), nullable=True)
    session_type = Column(String(16), nullable=False, default="regular")

    # Conviction tier carried over from the underlying signal (nullable for
    # rows recorded before tiering existed).
    conviction_tier = Column(String(24), nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Metadata
# Training runs, accuracy, feature importances
# ─────────────────────────────────────────────────────────────────────────────
class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64), nullable=False)
    ticker = Column(String(16), nullable=False, default="VOO", index=True)
    trained_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # JSON-encoded dict: {"feature_name": importance_value, ...}
    feature_importances = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# FCM Device Tokens
# Stores FCM registration tokens for push notification delivery.
# One row per physical device. Token is refreshed automatically by Firebase SDK.
# ─────────────────────────────────────────────────────────────────────────────
class DeviceToken(Base):
    __tablename__ = "device_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Full FCM registration token (up to ~512 chars)
    token = Column(String(512), nullable=False, unique=True, index=True)

    # Optional human-readable label (e.g. "Pixel 7", "Galaxy S24")
    device_name = Column(String(128), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # ── Per-device notification preferences ──────────────────────────────────
    # Minimum confidence (0.0–1.0) required to fire a BUY notification.
    # Derived from the user's buyThreshold + notificationSensitivity on Android.
    min_buy_threshold = Column(Float, nullable=False, default=0.70)

    # Minimum confidence (0.0–1.0) required to fire a SELL notification.
    min_sell_threshold = Column(Float, nullable=False, default=0.70)

    # When False, skip push notifications for extended-hours signals.
    extended_hours_notifications = Column(Boolean, nullable=False, default=True)

    # When True, only high-conviction signals trigger push notifications.
    # Default False preserves each device's current behavior (all signals).
    high_conviction_only = Column(Boolean, nullable=False, default=False)
