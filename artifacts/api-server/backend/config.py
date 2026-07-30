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

    # ── Ops / admin ───────────────────────────────────────────────────────────
    # Token required (X-Admin-Token header) for operator-only endpoints such
    # as resetting the persisted ML-fallback history. Falls back to
    # SESSION_SECRET when unset; if neither is set, admin endpoints are
    # disabled (503) rather than left open.
    ADMIN_TOKEN: str = ""
    SESSION_SECRET: str = ""

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

    # ── Notification reliability gate ─────────────────────────────────────────
    # Push notifications are suppressed when recent trade-cycle reliability is
    # poor: at least NOTIFY_RELIABILITY_MIN_CYCLES completed cycles must exist
    # in NOTIFY_RELIABILITY_WINDOW before the win-rate gate applies (so a fresh
    # system with no history is never muted), and the win rate must be at or
    # above NOTIFY_MIN_WIN_RATE for alerts to fire.
    NOTIFY_MIN_WIN_RATE: float = 0.40
    NOTIFY_RELIABILITY_MIN_CYCLES: int = 5
    NOTIFY_RELIABILITY_WINDOW: str = "30d"

    # ── Decision-layer filter thresholds ──────────────────────────────────────
    # Minimum cycle_quality_score required for a BUY signal to be emitted.
    # Cycle quality combines volatility_regime, gap_type, liquidity_class, and
    # confidence_momentum. SELL signals are allowed regardless of this score.
    DECISION_BUY_MIN_CYCLE_QUALITY: float = 0.6

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

    # ── Cross-bar spike detection ─────────────────────────────────────────────
    # A candle whose close deviates more than this fraction from the rolling
    # median of its neighbours is flagged as a cross-bar spike and quarantined.
    # Checked against a centered window of 5 bars (min 3 valid neighbours).
    # Normal intraday VOO 5-min moves stay well below this threshold; a 10 %
    # jump in a single bar is characteristic of a vendor data glitch.
    SPIKE_CLOSE_THRESHOLD: float = 0.10  # 10 % deviation from rolling median

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
