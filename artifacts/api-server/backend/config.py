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
    SPX_FUTURES_TICKER: str = "ES=F"   # S&P 500 E-mini futures (yfinance)
    # Warn when the latest stored SPX futures candle lags the latest VOO
    # trading day by more than this many trading days (staleness check).
    SPX_STALENESS_MAX_LAG_DAYS: int = 3
    # Warn when the latest stored VIX daily candle lags the latest VOO
    # trading day by more than this many trading days (staleness check).
    VIX_STALENESS_MAX_LAG_DAYS: int = 3
    # Warn when the latest stored VOO 5-min bar is older than this many
    # minutes while the regular market session is open (staleness check).
    FIVEMIN_STALENESS_MAX_AGE_MINUTES: int = 20

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
    # Additive gap-magnitude classification (does not affect gap_type)
    MICRO_GAP_THRESHOLD: float = 0.1  # % – |gap| below this is 'micro'
    MACRO_GAP_THRESHOLD: float = 1.0  # % – |gap| above this is 'macro'
    # Gap follow-through (momentum) influence on the short gauge (additive).
    # |gap_momentum| must exceed this % move before it affects the score:
    GAP_MOMENTUM_THRESHOLD: float = 0.1   # %
    # Score points added toward the gap direction on follow-through, or
    # away from it on a fading gap:
    GAP_MOMENTUM_SCORE_BOOST: float = 10.0

    # ── Macro override flag (ML feature, in-memory only) ─────────────────────
    # macro_override_flag = 1.0 when either:
    #   - volatility_regime == 'macro_shock', or
    #   - VIX regime >= MACRO_OVERRIDE_VIX_REGIME AND the absolute overnight
    #     move exceeds MACRO_OVERRIDE_OVERNIGHT_MOVE_PCT (%)
    # Backtested against real VOO/VIX history (scripts/backtest_macro_override.py):
    # HIGH catches CPI-surprise shock days (e.g. 2022-09-13) that EXTREME missed,
    # with zero false fires in calm 2017 / 2023 periods.
    MACRO_OVERRIDE_VIX_REGIME: str = "HIGH"
    MACRO_OVERRIDE_OVERNIGHT_MOVE_PCT: float = 2.0  # % overnight move

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
