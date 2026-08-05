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
    # Alert when the most recent daily VOO candle is more than this many
    # trading days old — the ingestion pipeline may have silently stopped.
    DAILY_CANDLE_STALE_THRESHOLD_DAYS: int = 3

    # ── Time-decay lambdas ────────────────────────────────────────────────────
    # Weight(t) = exp(-lambda * age)
    LAMBDA_LONG: float = 0.005   # per day   – long-trend gauge
    LAMBDA_SHORT: float = 0.05   # per minute – short-trend gauge

    # ── Signal thresholds ─────────────────────────────────────────────────────
    # Long-gauge max raw score: indicator (±30) + ML (±40) = ±70.
    # Thresholds are set at ±65 so a strong setup (ADX trending, both SMA/MACD
    # bullish, ML prediction ≥ ~0.94) can cross into actionable territory while
    # fresh data (time-decay weight ≈ 1.0).  Previously at ±70 the BUY/SELL
    # paths were mathematically unreachable even for perfect inputs.
    LONG_BUY_THRESHOLD: float = 65.0
    LONG_SELL_THRESHOLD: float = -65.0
    # Short-gauge max raw score:
    #   Regular  hours: indicator (±26) + ML (±40) = ±66 → threshold ±50 reachable
    #   Extended hours: indicator (±13) + ML (±40) = ±53 → threshold ±50 reachable
    # Previously at ±60, extended-hours was unreachable (double-penalty: indicator
    # contributions already halved AND base_weight = 0.5 → max extended = 26.5).
    # The base_weight is now 1.0 for all sessions (see short_gauge.py); ±50 lets
    # strong-but-not-perfect setups qualify without requiring unusually rare
    # full-agreement across all four indicators simultaneously.
    SHORT_BUY_THRESHOLD: float = 50.0
    SHORT_SELL_THRESHOLD: float = -50.0

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
    SPIKE_CLOSE_THRESHOLD: float = 0.10  # 10 % deviation from rolling median (intraday)

    # For daily candles, real macro events (COVID March 2020, post-CPI 2022)
    # can produce legitimate 5–9 % single-day moves, so the daily threshold is
    # set higher than the intraday threshold to avoid quarantining real data.
    # Historical VOO daily moves: COVID (-12 % on 2020-03-16), CPI shock
    # (-4.3 % on 2022-09-13) — a 12 % threshold catches true data glitches
    # while leaving room above the largest recorded real daily move.
    DAILY_SPIKE_CLOSE_THRESHOLD: float = 0.12  # 12 % deviation from rolling median (daily)

    # When this many cross-bar spike quarantines accumulate within a single
    # trading session (in-memory counter, reset each trading day) a WARN-level
    # structured log is emitted so operators know about a possible systematic
    # feed problem (broken source or unapplied split/adjustment).
    SPIKE_QUARANTINE_ALERT_COUNT: int = 3

    # Path to a small JSON file that persists the session spike-quarantine
    # counter across server restarts.  The file is written after every
    # quarantine and read on startup so a mid-session restart does not reset
    # the running total.  Set to "" to disable persistence (in-memory only).
    SPIKE_QUARANTINE_STATE_FILE: str = "spike_quarantine_state.json"

    # ── Data history ──────────────────────────────────────────────────────────
    HISTORY_YEARS: int = 10

    # ── Long-model target ─────────────────────────────────────────────────────
    # The long model is trained on meaningful 21-trading-day moves.  Returns
    # inside this band are treated as noise and excluded from the directional
    # classifier instead of being forced into BUY/SELL labels.
    LONG_LABEL_HORIZON_DAYS: int = 21
    LONG_MEANINGFUL_MOVE_THRESHOLD: float = 0.02

    # Target type for the long-trend model.  Supported values:
    #   direction      — binary BUY/SELL classifier (default, production)
    #   drawdown_event — binary classifier predicting a significant intra-horizon
    #                    drawdown; promoted when PR-AUC lift >= 2× AND
    #                    precision lift >= 2× on purged OOS evaluation.
    #   three_state    — three-class classifier (risk-off / neutral / risk-on);
    #                    promoted when macro-F1 > 0.40 AND each class F1 > 0.25.
    # Changing this puts the current model pkl into baseline mode until a
    # gate-passing retrain for the new target is completed.
    LONG_TARGET_TYPE: str = "direction"

    # Horizon and depth threshold for the drawdown_event target.
    LONG_DRAWDOWN_HORIZON: int = 21
    LONG_DRAWDOWN_THRESHOLD: float = 0.05  # 5 % intra-horizon drop

    # Horizon and return threshold for the three_state target neutral band.
    LONG_THREE_STATE_HORIZON: int = 21
    LONG_THREE_STATE_THRESHOLD: float = 0.02

    # ── Baseline-mode duration alert ──────────────────────────────────────────
    # Fire a one-time operator alert when the long-trend model stays in baseline
    # mode (no gate-passing trained model) for at least this many calendar days.
    # Set to 0 to disable the alert.
    LONG_BASELINE_MODE_ALERT_DAYS: int = 14

    LONG_MIN_TRAINING_ROWS: int = 80
    # A directional model must beat the majority-class baseline on honest
    # purged OOS evaluation before it can replace the active model.
    LONG_MIN_OOS_ACCURACY_LIFT: float = 0.0

    # ── Broader market context (experimental; gated behind OOS viability) ─────
    # Ablation control: False (default) = existing 19-feature set; True = add
    # 8 broader-context features (4 values + 4 freshness flags).  Only enable
    # after a candidate model with the new features clears the unchanged OOS
    # promotion gate.  Toggling this changes FEATURE_NAMES length, which puts
    # the current model pkl into baseline mode until the next retrain.
    LONG_BROADER_CONTEXT_ENABLED: bool = False
    # Data older than this many trading days is treated as stale; the
    # corresponding _missing indicator fires (= 1.0) so the model can learn
    # to discount absent context.  Two extra calendar days are added internally
    # to absorb weekends without penalising a Friday close on Monday.
    LONG_CONTEXT_STALENESS_MAX_DAYS: int = 5

    # Auto-promotion: when True, the server flips LONG_BROADER_CONTEXT_ENABLED
    # to True in-memory immediately after a gate-passing ablation so the next
    # scheduled retrain trains the 27-feature model without a manual config
    # change.  The promotion is also written to
    # ml/models/broader_context_promotion.json for operator audit and healthz
    # visibility.  NOTE: the in-memory flip does NOT survive a server restart
    # unless LONG_BROADER_CONTEXT_ENABLED is also set to True in the environment.
    # An FCM push notification is sent when the gate passes for the first time,
    # regardless of this switch.
    LONG_BROADER_CONTEXT_AUTO_ENABLE: bool = False

    # Tickers for broader context sources (ingested separately from VOO/VIX).
    # Empty string → that source is permanently absent (neutral fallback fires).
    VIX_SHORT_TICKER: str = "^VIX9D"   # 9-day VIX (term-structure numerator)
    VIX_LONG_TICKER: str = "^VIX3M"    # 3-month VIX (term-structure denominator)
    RATES_TICKER: str = "^TNX"          # 10-year Treasury yield (value ÷ 10 = %)
    CREDIT_HY_TICKER: str = "HYG"       # iShares iBoxx HY Corporate Bond ETF
    CREDIT_IG_TICKER: str = "LQD"       # iShares iBoxx IG Corporate Bond ETF
    BREADTH_TICKER: str = "^NYAD"       # NYSE Advance-Decline line

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton instance used throughout the application
settings = Settings()
