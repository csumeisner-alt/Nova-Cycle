# NovaCycle

AI-powered VOO ETF trading signal system. Dual long-trend and short-trend ML gauges, confidence history, filtered BUY/SELL signals, signal story cards, reliability metrics, and optional FCM push notifications.

---

## Architecture

```
NovaCycle
├── artifacts/api-server/backend/   ← Python FastAPI backend (running on Replit)
└── android/                        ← Kotlin/Jetpack Compose Android app
```

---

## Download the APK automatically

The easiest way to get the app on your phone is via the **GitHub Actions** workflow.

1. Push this repo to GitHub (see below for the one-line command).
2. Open the repo on GitHub → **Actions** tab → **Build NovaCycle APK**.
3. Click the latest successful run, then download the **novacyle-debug-apk** artifact.
4. Transfer the APK to your Android phone and install it.

No Firebase account, no Android Studio, and no command-line setup required.  
Detailed instructions: [`android/README_BUILD.md`](android/README_BUILD.md)

---

## Backend (FastAPI)

### Running on Replit
The API server starts automatically via the **API Server** workflow. After boot:
- Tables are created (SQLite, `novacycle.db`)
- Historical VOO + VIX data is fetched (10 years daily, 60 days 5-min)
- Both ML models are trained if no saved checkpoint exists
- APScheduler fires every 5 min (market hours) for incremental updates

### Key Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/healthz` | Health check + ticker info |
| POST | `/api/predict_long` | Long-trend gauge score + BUY/SELL |
| POST | `/api/predict_short` | Short-trend gauge score + BUY/SELL |
| GET | `/api/hold_time_estimate` | AI hold-time prediction |
| GET | `/api/confidence_history` | Long + short confidence over time |
| GET | `/api/signal_history` | Raw signal history |
| GET | `/api/filtered_signal_history` | Strongest-confidence filtered signals |
| GET | `/api/trade_history` | Completed BUY→SELL trade cycles + reliability metrics |
| GET | `/api/voo_candles` | VOO OHLCV candles |
| GET | `/api/vix_candles` | VIX candles |
| GET | `/api/indicators` | Live technical indicators |
| GET | `/api/gap_status` | Gap detection status |
| GET | `/api/model_metadata` | ML model training history |
| GET | `/docs` | Interactive Swagger UI |

### Signal Engine
- **Long gauge** (`/predict_long`): indicator score ±30, ML (XGBoost) score ±40, time-decay λ=0.005/day → BUY if > 70, SELL if < −70
- **Short gauge** (`/predict_short`): RSI/Stochastic/Bollinger scores, gap ±10, liquidity filter (×0.5 + threshold×1.25), neural-net MLP score, time-decay λ=0.05/min → BUY if > 60, SELL if < −60
- **Macro safety layer**: suppresses short BUY when long < −70 (and vice versa) unless ML > 80%

### Filtered Signal Algorithm
1. Sort all signals by timestamp
2. Group consecutive same-type signals
3. Keep only the highest-confidence signal per group
4. Enforce strict BUY → SELL → BUY alternation
5. Assign `cycle_id` (UUID) to each matched BUY→SELL pair

### ML Models
| Model | Type | Features | Target | Saved As |
|-------|------|----------|--------|----------|
| Long-trend | XGBoost | 14 daily indicators + time-decay weights | 1% daily gain in next 5 days | `ml/models/long_trend_model.pkl` |
| Short-trend | MLPClassifier (128→64→32) | 18 5-min features incl. gap, liquidity, overnight return | 0.3% gain in next 1 hour | `ml/models/short_trend_model.pkl` |

### Configuration (`config.py`)
All settings can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FCM_SERVER_KEY` | `""` | Firebase Cloud Messaging v1 service account key JSON |
| `LAMBDA_LONG` | `0.005` | Time-decay per day (long gauge) |
| `LAMBDA_SHORT` | `0.05` | Time-decay per minute (short gauge) |
| `LONG_BUY_THRESHOLD` | `70.0` | Score above this = BUY |
| `SHORT_BUY_THRESHOLD` | `60.0` | Score above this = BUY |

> **Note:** Do NOT set `DATABASE_URL` in the environment — the backend always uses SQLite (`novacycle.db`) regardless of any env var, to avoid conflicts with Replit's PostgreSQL `DATABASE_URL`.

### FCM Push Notifications (optional)
1. Download your Firebase service account JSON from Firebase Console → Project Settings → Service Accounts
2. Set the entire JSON as the `FCM_SERVER_KEY` environment variable (or Replit Secret)
3. The backend will send BUY/SELL alerts via FCM v1 HTTP API
4. On the Android side, add `google-services.json` and re-enable the commented Firebase lines (see `android/README_BUILD.md`)

---

## Android App

> **Full build instructions: [`android/README_BUILD.md`](android/README_BUILD.md)**

### Quick Start — GitHub Actions APK
1. Push this repo to a GitHub repository.
2. GitHub Actions automatically builds the APK on every push.
3. Download the APK from the Actions page and install it on your phone.

### Local Build
1. Open `android/` in Android Studio (File → Open → select the `android/` folder).
2. Wait for Gradle sync.
3. Build → Build APK(s) → locate `app-debug.apk`.

### App Screens
| Screen | Description |
|--------|-------------|
| **Dual Gauge** | Semicircular spring-animated gauges, hold-time card, macro override alert, 5-min auto-refresh |
| **Raw Chart** | Zoomable Canvas candlestick chart, 7 signal marker types, tap → Signal Story Card |
| **Filtered Chart** | Strongest-confidence signals only, trade-cycle shading, confidence momentum ribbon |
| **Confidence History** | Long + short confidence curves, EMA smoothing toggle, extended-hours shading |
| **Indicators** | Live indicator cards: RSI, Stochastic, StochRSI, MACD, MAs, Bollinger, CCI, Williams %R, ATR, ADX, VIX regime |
| **Hold Time** | AI-estimated position duration with confidence bar and reasoning bullets |
| **Settings** | BUY/SELL thresholds, weighting mode, smoothing, story card level, notification sensitivity |
| **Reliability** | Trade-cycle win-rate metrics, sortable/filterable BUY→SELL cycle table |

### Signal Story Card Detail Levels
- **Simple**: 3–4 bullets summarising why the signal fired
- **Advanced**: full indicator breakdown table, ML confidence bar, gap/liquidity info
- **Expert**: all of the above + session type, extended hours flag, macro override reason, time-decay formula

### Sensitivity Settings (presentation-layer only)
All sensitivity controls filter signals **client-side after receiving them from the backend**. No backend restart needed when you change settings.

---

## Notes & Limitations

- **Yahoo Finance in sandbox**: yfinance may fail in some hosted environments due to rate-limiting. The backend handles this gracefully; retry logic will pick up data on the next scheduled tick.
- **VOO only**: multi-ticker support is architecturally stubbed (`ticker` column on all tables) but not yet wired to a ticker selector — currently hardcoded to `VOO`.
- **Extended hours**: the short gauge applies a 0.5× weight multiplier to extended-hours bars; extended-hours signals are off by default in the Android app (toggle in Settings).
- **Not financial advice**: NovaCycle is a research and educational project. Signal outputs are not financial advice and do not constitute a recommendation to buy or sell any security.
