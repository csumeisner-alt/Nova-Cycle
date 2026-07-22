"""
NovaCycle Configuration
=======================
Centralised settings loaded from environment variables (or .env file).

NOTE: "Model currently trained only for ticker='VOO'. Multi-ticker support will be added later."
NOTE: "Pipeline currently fetches only VOO. Multi-ticker ingestion will be added later."
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./novacycle.db"

    # ── Ticker ────────────────────────────────────────────────────────────────
    TICKER: str = "VOO"
    VIX_TICKER: str = "^VIX"

    # ── Time-decay lambdas ────────────────────────────────────────────────────
    # Weight(t) = exp(-lambda * age)
    LAMBDA_LONG: float = 0.005   # per day   – long-trend gauge
    LAMBDA_SHORT: float = 0.05   # per minute – short-trend gauge

    # ── Signal thresholds ─────────────────────────────────────────────────────
    LONG_BUY_THRESHOLD: float = 70.0
    LONG_SELL_THRESHOLD: float = -70.0
    SHORT_BUY_THRESHOLD: float = 60.0
    SHORT_SELL_THRESHOLD: float = -60.0

    # ── Liquidity ─────────────────────────────────────────────────────────────
    # LiquidityScore = Volume_extended / Volume_regular
    # If < threshold: weights * 0.5, thresholds * 1.25, suppress weak signals
    LIQUIDITY_SCORE_THRESHOLD: float = 0.15

    # ── Gap detection ─────────────────────────────────────────────────────────
    # GapPercent = (PreMarketOpen - PreviousClose) / PreviousClose * 100
    GAP_UP_THRESHOLD: float = 1.0    # %
    GAP_DOWN_THRESHOLD: float = -1.0  # %

    # ── Firebase Cloud Messaging ───────────────────────────────────────────────
    FCM_SERVER_KEY: str = ""

    # ── Data history ──────────────────────────────────────────────────────────
    HISTORY_YEARS: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton instance used throughout the application
settings = Settings()
