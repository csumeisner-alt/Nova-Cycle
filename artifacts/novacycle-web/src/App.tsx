import { QueryClient, QueryClientProvider, useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Activity, Server, Clock, Download, ExternalLink, Terminal, AlertTriangle, CheckCircle2, RotateCcw, KeyRound, X, Minus, Gauge, TrendingUp, TrendingDown, Rss } from 'lucide-react';
import { PredictionCard } from '@/components/PredictionCard';
import { PerformanceDashboard } from '@/components/PerformanceDashboard';
import { TierTrackRecordPanel } from '@/components/TierTrackRecordPanel';
import { SignalHistoryPanel } from '@/components/SignalHistoryPanel';
import { TrainingStuckBanner } from '@/components/TrainingStuckBanner';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import { useState, useEffect } from 'react';

const queryClient = new QueryClient();

// Add a scanline effect via a fixed overlay
function Scanlines() {
  return (
    <div className="pointer-events-none fixed inset-0 z-50 opacity-[0.03] scanline mix-blend-overlay" />
  );
}

type RegimeEntry = {
  regime: string;
  regime_code: number;
  oos_samples: number;
  oos_accuracy: number;
  majority_baseline_accuracy: number;
  accuracy_lift_vs_majority: number;
  oos_brier_score: number;
  positive_rate: number;
};

type CalibrationReport = {
  evaluated?: boolean;
  calibrated?: boolean;
  reason?: string;
  oos_accuracy?: number | null;
  oos_brier_score?: number | null;
  oos_samples?: number | null;
  generated_at?: string | null;
  regime_breakdown?: RegimeEntry[];
};

type ModelHealth = {
  neutral_fallback?: boolean;
  last_training_success?: boolean | null;
  last_retrain_outcome?: 'success' | 'rolled_back' | 'failed' | null;
  last_retrain_rolled_back?: boolean;
  last_retrain_attempted_accuracy?: number | null;
  last_training_error?: string | null;
  last_training_attempted_at?: string | null;
  active_model_accuracy?: number | null;
  ml_fallback_count?: number;
  ml_fallback_total_count?: number;
  ml_fallback_total_last_at?: string | null;
  ml_fallback_total_last_reason?: string | null;
  ml_fallback_last_at?: string | null;
  ml_fallback_last_reason?: string | null;
  calibration?: CalibrationReport | null;
  walk_forward?: CalibrationReport | null;
};

const RECENT_FALLBACK_WINDOW_MS = 24 * 60 * 60 * 1000;

function isRecentFallback(lastAt: string | null | undefined): boolean {
  if (!lastAt) return false;
  const t = new Date(lastAt).getTime();
  return Number.isFinite(t) && Date.now() - t < RECENT_FALLBACK_WINDOW_MS;
}

function fmtAcc(v: number | null | undefined): string {
  return typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '—';
}

function RetrainOutcomeBadge({ outcome }: { outcome: ModelHealth['last_retrain_outcome'] }) {
  if (outcome === 'success') {
    return (
      <span className="inline-flex items-center space-x-1 text-primary bg-primary/10 border border-primary/20 px-2 py-0.5 rounded-full text-xs font-mono">
        <CheckCircle2 className="w-3 h-3" />
        <span>SUCCESS</span>
      </span>
    );
  }
  if (outcome === 'rolled_back') {
    return (
      <span className="inline-flex items-center space-x-1 text-amber-400 bg-amber-400/10 border border-amber-400/20 px-2 py-0.5 rounded-full text-xs font-mono">
        <RotateCcw className="w-3 h-3" />
        <span>ROLLED BACK</span>
      </span>
    );
  }
  if (outcome === 'failed') {
    return (
      <span className="inline-flex items-center space-x-1 text-destructive bg-destructive/10 border border-destructive/20 px-2 py-0.5 rounded-full text-xs font-mono">
        <AlertTriangle className="w-3 h-3" />
        <span>FAILED</span>
      </span>
    );
  }
  return <span className="text-xs font-mono text-muted-foreground">NO ATTEMPT YET</span>;
}

function RetrainStatusPanel({ health }: { health: any }) {
  const models: Record<string, ModelHealth> = health?.models ?? {};
  const entries = Object.entries(models);
  if (entries.length === 0) return null;

  return (
    <div className="mt-8 p-4 bg-white/[0.02] rounded-lg border border-white/5" data-testid="panel-retrain-status">
      <div className="flex items-center space-x-2 text-muted-foreground mb-4">
        <Activity className="w-4 h-4" />
        <span className="text-sm font-medium tracking-wide">MODEL RETRAIN STATUS</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {entries.map(([name, m]) => (
          <div
            key={name}
            className="p-3 bg-black/30 rounded-lg border border-white/5 space-y-2 font-mono text-sm"
            data-testid={`retrain-card-${name}`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs tracking-wide uppercase">{name.replace('_', ' ')}</span>
              <RetrainOutcomeBadge outcome={m.last_retrain_outcome ?? null} />
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-muted-foreground">Active model accuracy</span>
              <span data-testid={`text-active-accuracy-${name}`}>{fmtAcc(m.active_model_accuracy)}</span>
            </div>
            {m.last_retrain_outcome === 'rolled_back' && (
              <>
                <div className="flex items-baseline justify-between">
                  <span className="text-xs text-muted-foreground">Attempted accuracy (discarded)</span>
                  <span className="text-amber-400" data-testid={`text-attempted-accuracy-${name}`}>
                    {fmtAcc(m.last_retrain_attempted_accuracy)}
                  </span>
                </div>
                <p className="text-xs text-amber-200/70 leading-snug" data-testid={`text-rollback-reason-${name}`}>
                  {m.last_training_error || 'Retrain flagged; model restored to last known-good version.'}
                </p>
              </>
            )}
            {m.last_retrain_outcome === 'failed' && m.last_training_error && (
              <p className="text-xs text-destructive/80 leading-snug" data-testid={`text-fail-reason-${name}`}>
                {m.last_training_error}
              </p>
            )}
            {m.last_training_attempted_at && (
              <div className="text-[11px] text-muted-foreground">
                Last attempt: {new Date(m.last_training_attempted_at).toLocaleString('en-US', { hour12: false })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

const REGIME_COLORS: Record<string, string> = {
  LOW: 'text-sky-400',
  NORMAL: 'text-primary',
  HIGH: 'text-amber-400',
  EXTREME: 'text-destructive',
};

function RegimeBreakdownTable({
  breakdown,
  evaluated,
  reason,
  modelKey,
}: {
  breakdown: RegimeEntry[] | undefined;
  evaluated: boolean;
  reason: string | undefined;
  modelKey: string;
}) {
  if (!evaluated || !breakdown || breakdown.length === 0) {
    return (
      <div
        className="p-3 bg-black/30 rounded-lg border border-white/5 font-mono text-sm text-muted-foreground"
        data-testid={`regime-breakdown-empty-${modelKey}`}
      >
        {reason
          ? `Walk-forward evaluation not available: ${reason}`
          : 'No per-regime OOS data yet — walk-forward evaluation has not completed.'}
      </div>
    );
  }
  return (
    <div className="overflow-x-auto" data-testid={`regime-breakdown-table-${modelKey}`}>
      <table className="w-full font-mono text-sm border-collapse">
        <thead>
          <tr className="text-[11px] text-muted-foreground uppercase tracking-wide border-b border-white/5">
            <th className="text-left py-2 pr-4">Regime</th>
            <th className="text-right py-2 px-2">Samples</th>
            <th className="text-right py-2 px-2">OOS Acc</th>
            <th className="text-right py-2 px-2">Baseline</th>
            <th className="text-right py-2 px-2">Lift</th>
            <th className="text-right py-2 pl-2">Brier</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.04]">
          {breakdown.map((row) => {
            const lift = row.accuracy_lift_vs_majority;
            const liftColor = lift > 0.01 ? 'text-primary' : lift < -0.01 ? 'text-destructive' : 'text-muted-foreground';
            const regimeColor = REGIME_COLORS[row.regime] ?? 'text-foreground';
            const isVolatileRegime = row.regime === 'HIGH' || row.regime === 'EXTREME';
            const isDegraded = isVolatileRegime && lift < 0;
            return (
              <tr
                key={row.regime_code}
                className={`transition-colors ${isDegraded ? 'bg-destructive/10 hover:bg-destructive/15' : 'hover:bg-white/[0.02]'}`}
                data-testid={`regime-row-${modelKey}-${row.regime.toLowerCase()}`}
                aria-label={isDegraded ? `${row.regime} regime degraded under volatility stress` : undefined}
              >
                <td className={`py-2.5 pr-4 font-medium ${regimeColor}`}>
                  <span className="flex items-center gap-1.5">
                    {row.regime}
                    {isDegraded && (
                      <AlertTriangle
                        className="w-3.5 h-3.5 text-destructive shrink-0"
                        aria-hidden="true"
                        data-testid={`icon-regime-degraded-${modelKey}-${row.regime.toLowerCase()}`}
                      />
                    )}
                  </span>
                </td>
                <td className="py-2.5 px-2 text-right text-muted-foreground">
                  {row.oos_samples.toLocaleString()}
                </td>
                <td className="py-2.5 px-2 text-right" data-testid={`regime-acc-${modelKey}-${row.regime.toLowerCase()}`}>
                  {(row.oos_accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-2 text-right text-muted-foreground">
                  {(row.majority_baseline_accuracy * 100).toFixed(1)}%
                </td>
                <td className={`py-2.5 px-2 text-right font-medium ${liftColor}`} data-testid={`regime-lift-${modelKey}-${row.regime.toLowerCase()}`}>
                  {lift >= 0 ? '+' : ''}{(lift * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 pl-2 text-right text-muted-foreground">
                  {row.oos_brier_score.toFixed(4)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-3 text-[11px] text-muted-foreground font-mono leading-relaxed">
        Lift = OOS accuracy − majority-class baseline. Positive lift means the model beats always-predicting-the-majority class.
        HIGH/EXTREME rows with negative lift indicate the model degrades under volatility stress.
      </p>
    </div>
  );
}

function RegimeBreakdownPanel({ health }: { health: any }) {
  const models: Record<string, ModelHealth> = health?.models ?? {};
  const longTrend = models['long_trend'];
  const shortTrend = models['short_trend'];

  // Only render if at least one model is present
  if (!longTrend && !shortTrend) return null;

  const longCalibration = longTrend?.calibration;
  const shortWalkForward = shortTrend?.walk_forward;

  return (
    <div
      className="mt-8 p-4 bg-white/[0.02] rounded-lg border border-white/5"
      data-testid="panel-regime-breakdown"
    >
      <div className="flex items-center space-x-2 text-muted-foreground mb-4">
        <Gauge className="w-4 h-4" />
        <span className="text-sm font-medium tracking-wide">OOS BREAKDOWN BY VIX REGIME</span>
      </div>

      <div className="space-y-6">
        {longTrend && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-muted-foreground uppercase tracking-wide">Long-Trend Model</span>
              {longCalibration?.generated_at && (
                <span className="text-[11px] font-mono text-muted-foreground hidden sm:block">
                  {new Date(longCalibration.generated_at).toLocaleString('en-US', { hour12: false })}
                </span>
              )}
            </div>
            <RegimeBreakdownTable
              breakdown={longCalibration?.regime_breakdown}
              evaluated={longCalibration?.evaluated ?? false}
              reason={longCalibration?.reason}
              modelKey="long-trend"
            />
          </div>
        )}

        {shortTrend && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-muted-foreground uppercase tracking-wide">Short-Trend Model</span>
              {shortWalkForward?.generated_at && (
                <span className="text-[11px] font-mono text-muted-foreground hidden sm:block">
                  {new Date(shortWalkForward.generated_at).toLocaleString('en-US', { hour12: false })}
                </span>
              )}
            </div>
            <RegimeBreakdownTable
              breakdown={shortWalkForward?.regime_breakdown}
              evaluated={shortWalkForward?.evaluated ?? false}
              reason={shortWalkForward?.reason}
              modelKey="short-trend"
            />
          </div>
        )}
      </div>
    </div>
  );
}


function FallbackHistoryPanel({ health }: { health: any }) {
  const qc = useQueryClient();
  const [showPrompt, setShowPrompt] = useState(false);
  const [token, setToken] = useState('');
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const models: Record<string, ModelHealth> = health?.models ?? {};
  const totalPersisted = Object.values(models).reduce((s, m) => s + (m.ml_fallback_total_count ?? 0), 0);
  const totalSinceStartup = Object.values(models).reduce((s, m) => s + (m.ml_fallback_count ?? 0), 0);
  const lastResetAt: string | null = health?.fallback_stats_last_reset_at ?? null;

  const resetMutation = useMutation({
    mutationFn: async (adminToken: string) => {
      const res = await fetch('/api/admin/reset_fallback_stats', {
        method: 'POST',
        headers: { 'X-Admin-Token': adminToken },
      });
      if (!res.ok) {
        let detail = '';
        try {
          const body = await res.json();
          detail = body?.detail ?? '';
        } catch {
          /* ignore */
        }
        if (res.status === 403) {
          throw new Error(detail || 'Invalid admin token. Check the token and try again.');
        }
        if (res.status === 503) {
          throw new Error(detail || 'Admin endpoints are disabled on the backend (no ADMIN_TOKEN or SESSION_SECRET configured).');
        }
        throw new Error(detail || `Reset failed (HTTP ${res.status})`);
      }
      return res.json();
    },
    onSuccess: (data) => {
      setShowPrompt(false);
      setToken('');
      setSuccessMsg(`Fallback history cleared at ${new Date(data.reset_at).toLocaleTimeString('en-US', { hour12: false })}`);
      qc.invalidateQueries({ queryKey: ['healthz'] });
    },
  });

  const openPrompt = () => {
    setSuccessMsg(null);
    resetMutation.reset();
    setShowPrompt(true);
  };

  return (
    <div className="mt-8 p-4 bg-white/[0.02] rounded-lg border border-white/5" data-testid="panel-fallback-history">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2 text-muted-foreground">
          <RotateCcw className="w-4 h-4" />
          <span className="text-sm font-medium tracking-wide">FALLBACK HISTORY</span>
        </div>
        {!showPrompt && (
          <button
            onClick={openPrompt}
            data-testid="button-clear-fallback"
            className="inline-flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-md border border-destructive/30 text-destructive bg-destructive/5 hover:bg-destructive/15 transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Clear fallback history</span>
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-sm">
        <div className="space-y-1">
          <div className="text-xs text-muted-foreground">Persisted fallbacks</div>
          <div className="text-lg" data-testid="text-fallback-total">{totalPersisted}</div>
        </div>
        <div className="space-y-1">
          <div className="text-xs text-muted-foreground">Since startup</div>
          <div className="text-lg" data-testid="text-fallback-startup">{totalSinceStartup}</div>
        </div>
        <div className="space-y-1">
          <div className="text-xs text-muted-foreground">Last reset</div>
          <div className="text-sm pt-1" data-testid="text-fallback-last-reset">
            {lastResetAt ? new Date(lastResetAt).toLocaleString('en-US', { hour12: false }) : 'Never'}
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        {Object.entries(models).map(([name, m]) => {
          const count = m.ml_fallback_total_count ?? 0;
          const lastAt = m.ml_fallback_total_last_at ?? null;
          const lastReason = m.ml_fallback_total_last_reason ?? null;
          const recent = isRecentFallback(lastAt);
          return (
            <div
              key={name}
              data-testid={`row-fallback-model-${name}`}
              className={`p-3 rounded-lg border font-mono text-sm ${
                recent
                  ? 'border-destructive/40 bg-destructive/10'
                  : 'border-white/5 bg-white/[0.02]'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-xs tracking-wide uppercase">{name.replace('_', ' ')}</span>
                  {recent && (
                    <span
                      data-testid={`badge-fallback-recent-${name}`}
                      className="inline-flex items-center space-x-1 text-[10px] px-1.5 py-0.5 rounded bg-destructive/20 text-destructive"
                    >
                      <AlertTriangle className="w-3 h-3" />
                      <span>RECENT FALLBACK</span>
                    </span>
                  )}
                </div>
                <span className="text-xs text-muted-foreground">
                  <span data-testid={`text-fallback-count-${name}`} className={count > 0 ? 'text-foreground' : ''}>
                    {count}
                  </span>{' '}
                  fallback{count === 1 ? '' : 's'}
                </span>
              </div>
              <div className="mt-1.5 flex flex-col sm:flex-row sm:items-center sm:space-x-4 space-y-0.5 sm:space-y-0 text-[11px] text-muted-foreground">
                <span data-testid={`text-fallback-last-at-${name}`}>
                  Last: {lastAt ? new Date(lastAt).toLocaleString('en-US', { hour12: false }) : '—'}
                </span>
                <span data-testid={`text-fallback-last-reason-${name}`} className={recent ? 'text-destructive/90' : ''}>
                  Reason: {lastReason ?? '—'}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {showPrompt && (
        <form
          className="mt-4 p-3 bg-black/40 rounded-lg border border-white/10 space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            if (token.trim()) resetMutation.mutate(token.trim());
          }}
        >
          <div className="flex items-center justify-between text-xs text-muted-foreground font-mono">
            <div className="flex items-center space-x-2">
              <KeyRound className="w-3.5 h-3.5" />
              <span>Enter admin token to clear the persisted fallback history</span>
            </div>
            <button
              type="button"
              aria-label="Cancel"
              onClick={() => {
                setShowPrompt(false);
                setToken('');
                resetMutation.reset();
              }}
              className="p-1 rounded hover:bg-white/10 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="flex items-center space-x-2">
            <input
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="X-Admin-Token"
              autoFocus
              data-testid="input-admin-token"
              className="flex-1 bg-background/80 border border-white/10 rounded-md px-3 py-1.5 font-mono text-sm focus:outline-none focus:ring-1 focus:ring-primary placeholder:text-muted-foreground/50"
            />
            <button
              type="submit"
              disabled={!token.trim() || resetMutation.isPending}
              data-testid="button-confirm-clear"
              className="text-xs font-mono px-3 py-1.5 rounded-md bg-destructive/80 text-white hover:bg-destructive transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {resetMutation.isPending ? 'Clearing…' : 'Confirm clear'}
            </button>
          </div>
          {resetMutation.isError && (
            <div className="flex items-start space-x-2 text-xs font-mono text-destructive" data-testid="text-reset-error">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>{resetMutation.error instanceof Error ? resetMutation.error.message : 'Reset failed'}</span>
            </div>
          )}
        </form>
      )}

      {successMsg && !showPrompt && (
        <div className="mt-4 flex items-center space-x-2 text-xs font-mono text-primary" data-testid="text-reset-success">
          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}
    </div>
  );
}


function OhlcQuarantinePanel({ health }: { health: any }) {
  const q: { count?: number; last_at?: string | null; last_ts?: string | null; last_reason?: string | null } =
    health?.ohlc_quarantine ?? {};
  const count = q.count ?? 0;

  return (
    <div
      className={`mt-8 p-4 rounded-lg border ${
        count > 0
          ? 'bg-amber-400/5 border-amber-400/25'
          : 'bg-white/[0.02] border-white/5'
      }`}
      data-testid="panel-ohlc-quarantine"
    >
      <div className="flex items-center space-x-2 mb-4">
        {count > 0 ? (
          <AlertTriangle className="w-4 h-4 text-amber-400" />
        ) : (
          <CheckCircle2 className="w-4 h-4 text-muted-foreground" />
        )}
        <span
          className={`text-sm font-medium tracking-wide ${count > 0 ? 'text-amber-400' : 'text-muted-foreground'}`}
        >
          OHLC DATA QUALITY
        </span>
        {count > 0 && (
          <span
            className="ml-auto inline-flex items-center text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-400/15 text-amber-400 border border-amber-400/30"
            data-testid="badge-quarantine-count"
          >
            {count} candle{count === 1 ? '' : 's'} quarantined
          </span>
        )}
      </div>

      {count > 0 ? (
        <div className="space-y-3 font-mono text-sm">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">Total quarantined</div>
              <div className="text-lg text-amber-400" data-testid="text-quarantine-count">{count}</div>
            </div>
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">Last candle timestamp</div>
              <div className="text-sm pt-1" data-testid="text-quarantine-last-ts">
                {q.last_ts ?? '—'}
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">Detected at</div>
              <div className="text-sm pt-1" data-testid="text-quarantine-last-at">
                {q.last_at ? new Date(q.last_at).toLocaleString('en-US', { hour12: false }) : '—'}
              </div>
            </div>
          </div>
          {q.last_reason && (
            <div
              className="p-3 bg-black/30 rounded-lg border border-amber-400/20 text-[11px] text-amber-200/80 leading-relaxed break-words"
              data-testid="text-quarantine-last-reason"
            >
              {q.last_reason}
            </div>
          )}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground font-mono" data-testid="text-quarantine-clean">
          No malformed candles detected since last startup.
        </p>
      )}
    </div>
  );
}

// ── Context-feed labels ────────────────────────────────────────────────────
const FEED_LABELS: Record<string, string> = {
  vix_short: 'VIX9D',
  vix_long: 'VIX3M',
  rates: 'TNX',
  credit_hy: 'HYG',
  credit_ig: 'LQD',
  breadth: 'NYAD',
};

type ContextFeedEntry = {
  ticker?: string;
  feed_key?: string;
  stale: boolean;
  lag_trading_days?: number | null;
  max_lag_trading_days?: number;
  detail?: string | null;
  [key: string]: unknown; // latest_<feed_key> dynamic key
};

function ContextFeedsPanel({ health }: { health: any }) {
  const feeds: ContextFeedEntry[] = Array.isArray(health?.context_feeds)
    ? health.context_feeds
    : [];

  if (feeds.length === 0) return null;

  const anyStale = feeds.some((f) => f.stale);

  return (
    <div
      className={`mt-8 p-4 rounded-lg border ${
        anyStale ? 'bg-amber-400/5 border-amber-400/25' : 'bg-white/[0.02] border-white/5'
      }`}
      data-testid="panel-context-feeds"
    >
      <div className="flex items-center space-x-2 mb-4">
        {anyStale ? (
          <AlertTriangle className="w-4 h-4 text-amber-400" />
        ) : (
          <Rss className="w-4 h-4 text-muted-foreground" />
        )}
        <span
          className={`text-sm font-medium tracking-wide ${
            anyStale ? 'text-amber-400' : 'text-muted-foreground'
          }`}
        >
          CONTEXT FEED HEALTH
        </span>
        {anyStale && (
          <span className="ml-auto inline-flex items-center text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-400/15 text-amber-400 border border-amber-400/30">
            {feeds.filter((f) => f.stale).length} stale
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 font-mono text-sm">
        {feeds.map((feed) => {
          const key = feed.feed_key ?? '';
          const label = FEED_LABELS[key] ?? feed.ticker ?? key;
          const latestKey = `latest_${key}`;
          const latestDate = feed[latestKey] as string | null | undefined;
          const lag = feed.lag_trading_days;
          const maxLag = feed.max_lag_trading_days;
          const stale = feed.stale;

          return (
            <div
              key={key || label}
              className={`p-3 rounded-lg border space-y-1.5 ${
                stale
                  ? 'bg-amber-400/5 border-amber-400/25'
                  : 'bg-black/30 border-white/5'
              }`}
              data-testid={`context-feed-${key}`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold tracking-wider uppercase">
                  {label}
                </span>
                {stale ? (
                  <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full bg-amber-400/15 text-amber-400 border border-amber-400/30">
                    <AlertTriangle className="w-2.5 h-2.5" />
                    STALE
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
                    <CheckCircle2 className="w-2.5 h-2.5" />
                    FRESH
                  </span>
                )}
              </div>

              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Latest</span>
                  <span
                    className={stale ? 'text-amber-300' : ''}
                    data-testid={`context-feed-latest-${key}`}
                  >
                    {latestDate ?? '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Lag</span>
                  <span
                    className={stale ? 'text-amber-300' : ''}
                    data-testid={`context-feed-lag-${key}`}
                  >
                    {lag != null
                      ? `${lag}d${maxLag != null ? ` / ${maxLag}d max` : ''}`
                      : '—'}
                  </span>
                </div>
              </div>

              {stale && feed.detail && (
                <p
                  className="text-[10px] text-amber-200/70 leading-snug pt-0.5 border-t border-amber-400/15"
                  data-testid={`context-feed-detail-${key}`}
                >
                  {feed.detail}
                </p>
              )}
            </div>
          );
        })}
      </div>

      {!anyStale && (
        <p className="mt-3 text-xs text-muted-foreground font-mono" data-testid="text-context-feeds-ok">
          All six broader-context feeds are current. Missing-feature flags will not fire.
        </p>
      )}
    </div>
  );
}

function PredictionsPanel() {
  return (
    <div className="mt-8 p-4 bg-white/[0.02] rounded-lg border border-white/5" data-testid="panel-predictions">
      <div className="flex items-center space-x-2 text-muted-foreground mb-4">
        <Gauge className="w-4 h-4" />
        <span className="text-sm font-medium tracking-wide">TRADING SIGNALS</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PredictionCard name="long" label="Long Trend" />
        <PredictionCard name="short" label="Short Trend" />
      </div>
    </div>
  );
}

type PriceSnapshot = {
  current_price?: number | null;
  current_timestamp?: string | null;
  current_session?: string | null;
  is_extended_hours?: boolean;
  day_change_percent?: number | null;
  day_direction?: 'up' | 'down' | 'flat';
};

function VooPriceStrip() {
  const { data, isLoading, isError } = useQuery<PriceSnapshot>({
    queryKey: ['price-snapshot', 'VOO'],
    queryFn: async () => {
      const res = await fetch('/api/price_snapshot?ticker=VOO');
      if (!res.ok) throw new Error(`Price unavailable (HTTP ${res.status})`);
      return res.json();
    },
    refetchInterval: 60_000,
    retry: 1,
  });

  const price = typeof data?.current_price === 'number' && Number.isFinite(data.current_price)
    ? data.current_price
    : null;
  const change = typeof data?.day_change_percent === 'number' && Number.isFinite(data.day_change_percent)
    ? data.day_change_percent
    : null;
  const direction = data?.day_direction;
  const DirectionIcon = direction === 'up' ? TrendingUp : direction === 'down' ? TrendingDown : Minus;
  const directionClass = direction === 'up'
    ? 'text-emerald-400'
    : direction === 'down'
      ? 'text-red-400'
      : 'text-muted-foreground';
  const session = data?.current_session?.replace('_', ' ') || 'latest feed';

  return (
    <div
      className="mt-8 flex items-center justify-between gap-4 rounded-lg border border-primary/15 bg-primary/[0.04] px-4 py-3"
      data-testid="voo-price-strip"
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-xs font-mono tracking-wide text-muted-foreground">
          <span className="text-primary">VOO</span>
          <span>LIVE PRICE</span>
          {data?.is_extended_hours && (
            <span className="rounded border border-amber-400/30 bg-amber-400/10 px-1.5 py-0.5 text-[10px] text-amber-300">
              EXTENDED
            </span>
          )}
        </div>
        <div className="mt-1 text-[11px] font-mono capitalize text-muted-foreground">
          {isLoading ? 'Reading market feed…' : isError ? 'Price feed unavailable' : `${session} · refreshes every minute`}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2 font-mono">
        <span className="text-xl text-foreground" data-testid="voo-current-price">
          {price == null ? '—' : `$${price.toFixed(2)}`}
        </span>
        <span className={directionClass} data-testid={`voo-day-direction-${direction ?? 'flat'}`}>
          <DirectionIcon className="h-5 w-5" aria-label={direction === 'down' ? 'Moving down today' : direction === 'up' ? 'Moving up today' : 'Flat today'} />
        </span>
        <span className={`text-xs ${directionClass}`} data-testid="voo-day-change">
          {change == null ? '—' : `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`}
        </span>
      </div>
    </div>
  );
}

function StatusDashboard() {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const { data: health, isLoading, isError, error } = useQuery({
    queryKey: ['healthz'],
    queryFn: async () => {
      const res = await fetch('/api/healthz');
      if (!res.ok) {
        throw new Error('Backend unreachable');
      }
      return res.json();
    },
    refetchInterval: 5000, // Check every 5 seconds
    retry: 2,
  });

  const isDegraded = !isLoading && !isError && health?.status === 'degraded';
  const degradedModels: string[] = health?.models
    ? Object.entries(health.models as Record<string, { neutral_fallback?: boolean; last_training_success?: boolean | null }>)
        .filter(([, m]) => m.neutral_fallback || m.last_training_success === false)
        .map(([name]) => name)
    : [];
  const alerts: string[] = Array.isArray(health?.alerts) ? health.alerts : [];

  return (
    <div className="min-h-[100dvh] w-full flex flex-col items-center justify-center relative overflow-hidden bg-background selection:bg-primary selection:text-primary-foreground">
      <Scanlines />
      
      {/* Background ambient glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/5 rounded-full blur-[120px] pointer-events-none" />

      <main className="relative z-10 w-full max-w-2xl px-6 py-12 flex flex-col items-center">
        
        {/* Header section */}
        <div className="flex flex-col items-center text-center space-y-4 mb-12">
          <div className="inline-flex items-center justify-center p-3 rounded-2xl bg-secondary mb-2 border border-primary/20 glow-primary">
            <Activity className="w-8 h-8 text-primary" strokeWidth={2} />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight glow-text-primary">
            NovaCycle API
          </h1>
          <p className="text-muted-foreground text-lg max-w-[400px]">
            Mission control for the NovaCycle AI-powered VOO ETF trading signal network.
          </p>
        </div>

        {/* Status Card */}
        <div className="w-full bg-card/40 backdrop-blur-md border border-card-border p-1 rounded-2xl shadow-2xl mb-8">
          <div className="bg-background/50 rounded-xl p-6 md:p-8">
            <div className="flex items-center justify-between mb-8 pb-6 border-b border-white/5">
              <div className="flex items-center space-x-3">
                <Server className="w-5 h-5 text-muted-foreground" />
                <h2 className="text-lg font-medium tracking-wide">SYSTEM STATUS</h2>
              </div>
              <div className="flex items-center space-x-2">
                {isLoading ? (
                  <div className="flex items-center space-x-2 text-muted-foreground bg-white/5 px-3 py-1 rounded-full text-sm">
                    <div className="w-2 h-2 rounded-full bg-muted animate-pulse" />
                    <span className="font-mono">Pinging...</span>
                  </div>
                ) : isError ? (
                  <div className="flex items-center space-x-2 text-destructive bg-destructive/10 px-3 py-1 rounded-full text-sm border border-destructive/20">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span className="font-mono font-medium">OFFLINE</span>
                  </div>
                ) : isDegraded ? (
                  <div className="flex items-center space-x-2 text-amber-400 bg-amber-400/10 px-3 py-1 rounded-full text-sm border border-amber-400/20">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span className="font-mono font-medium">DEGRADED</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2 text-primary bg-primary/10 px-3 py-1 rounded-full text-sm border border-primary/20">
                    <div className="w-2 h-2 rounded-full bg-primary animate-pulse glow-primary" />
                    <span className="font-mono font-medium">OPERATIONAL</span>
                  </div>
                )}
              </div>
            </div>

            {isDegraded && (
              <div className="mb-8 p-4 bg-amber-400/5 rounded-lg border border-amber-400/20" data-testid="banner-degraded">
                <div className="flex items-center space-x-2 text-amber-400 mb-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span className="font-mono font-medium text-sm">SYSTEM DEGRADED</span>
                </div>
                <p className="text-sm text-amber-200/80 mb-2">
                  {degradedModels.length > 0
                    ? `Predictions may be unreliable — affected model${degradedModels.length > 1 ? 's' : ''}: ${degradedModels.join(', ')}.`
                    : 'Some system components are degraded.'}
                </p>
                {alerts.length > 0 && (
                  <ul className="space-y-1 font-mono text-xs text-amber-200/60 list-disc list-inside">
                    {alerts.map((a, i) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <span className="text-sm text-muted-foreground flex items-center space-x-2">
                  <Terminal className="w-4 h-4" />
                  <span>Service Identifier</span>
                </span>
                <div className="font-mono text-lg truncate">
                  {isLoading ? (
                    <div className="h-7 w-32 bg-white/5 rounded animate-pulse" />
                  ) : isError ? (
                    <span className="text-muted-foreground">Unknown</span>
                  ) : (
                    health?.service || 'novacycle-api'
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-sm text-muted-foreground flex items-center space-x-2">
                  <Clock className="w-4 h-4" />
                  <span>Last Checked</span>
                </span>
                <div className="font-mono text-lg">
                  {now.toLocaleTimeString('en-US', { 
                    hour12: false, 
                    hour: '2-digit', 
                    minute: '2-digit', 
                    second: '2-digit',
                    fractionalSecondDigits: 2 
                  })}
                </div>
              </div>
            </div>
            
             <VooPriceStrip />
             <PredictionsPanel />
             <TierTrackRecordPanel />
             <SignalHistoryPanel />

            {!isLoading && !isError && health && <RetrainStatusPanel health={health} />}

            {!isLoading && !isError && health && <RegimeBreakdownPanel health={health} />}

            {!isLoading && !isError && health && <FallbackHistoryPanel health={health} />}

            {!isLoading && !isError && health && <OhlcQuarantinePanel health={health} />}

            {!isLoading && !isError && health && <ContextFeedsPanel health={health} />}

            {/* Raw JSON View */}
            {!isLoading && !isError && health && (
              <div className="mt-8 p-4 bg-black/40 rounded-lg border border-white/5 font-mono text-xs text-muted-foreground overflow-x-auto">
                <div className="flex items-center space-x-2 mb-2 text-white/30">
                  <Terminal className="w-3 h-3" />
                  <span>/api/healthz response</span>
                </div>
                <pre className="text-primary/70">{JSON.stringify(health, null, 2)}</pre>
              </div>
            )}
            
            {isError && (
              <div className="mt-8 p-4 bg-destructive/5 rounded-lg border border-destructive/10 font-mono text-xs text-destructive">
                Failed to fetch backend health status. The service might be deploying or temporarily unavailable.
                <br />
                <span className="text-muted-foreground mt-2 block">{error instanceof Error ? error.message : 'Unknown error'}</span>
              </div>
            )}
          </div>
        </div>

        {/* CTA Section */}
        <DownloadApkSection />

      </main>
    </div>
  );
}

type GitHubRelease = {
  tag_name: string;
  name: string | null;
  published_at: string | null;
  draft: boolean;
  prerelease: boolean;
  assets: { name: string; browser_download_url: string; size: number }[];
};

const RELEASES_API_URL = 'https://api.github.com/repos/csumeisner-alt/Nova-Cycle/releases?per_page=15';
const RELEASES_PAGE_URL = 'https://github.com/csumeisner-alt/Nova-Cycle/releases';

type ReleaseInfo = {
  tag_name: string;
  published_at: string | null;
  apk_url: string;
  stale?: boolean;
};

// Fallback path: query GitHub directly from the browser (subject to
// anonymous rate limits) when the backend proxy is unavailable.
async function fetchReleaseFromGitHub(): Promise<ReleaseInfo | null> {
  const res = await fetch(RELEASES_API_URL, {
    headers: { Accept: 'application/vnd.github+json' },
  });
  if (!res.ok) throw new Error(`GitHub API returned ${res.status}`);
  const releases: GitHubRelease[] = await res.json();
  // Skip the rolling "latest" alias and drafts/prereleases; pick the
  // newest *versioned* release so the link targets an immutable asset URL
  // that browser/CDN redirect caching can never turn stale.
  const versioned = releases
    .filter((r) => r.tag_name !== 'latest' && !r.draft && !r.prerelease)
    .filter((r) => r.assets.some((a) => a.name.toLowerCase().endsWith('.apk')))
    .sort((a, b) => (b.published_at ?? '').localeCompare(a.published_at ?? ''));
  const release = versioned[0];
  if (!release) return null;
  // Prefer the canonical CI asset name; fall back to any APK in the release.
  const apk =
    release.assets.find((a) => a.name === 'app-release.apk') ??
    release.assets.find((a) => a.name.toLowerCase().endsWith('.apk'));
  if (!apk) return null;
  return {
    tag_name: release.tag_name,
    published_at: release.published_at,
    apk_url: apk.browser_download_url,
    stale: false,
  };
}

function DownloadApkSection() {
  const { data: release, isLoading, isError } = useQuery({
    queryKey: ['latest-apk-release'],
    queryFn: async (): Promise<ReleaseInfo | null> => {
      // Primary path: backend proxy (/releases/latest) — it caches GitHub
      // responses server-side and serves the last known release when GitHub
      // rate-limits (marking the response stale: true in that case).
      try {
        const res = await fetch('/api/releases/latest');
        if (res.ok) {
          const data = await res.json();
          if (data?.ok && data?.release) {
            const r = data.release;
            const apk =
              (r.assets ?? []).find((a: { name: string; browser_download_url: string }) => a.name === 'app-release.apk') ??
              (r.assets ?? []).find((a: { name: string; browser_download_url: string }) => a.name.toLowerCase().endsWith('.apk'));
            if (apk) {
              return {
                tag_name: r.tag_name,
                published_at: r.published_at ?? null,
                apk_url: apk.browser_download_url,
                stale: Boolean(data.stale),
              };
            }
          }
        }
      } catch {
        // fall through to direct GitHub fetch
      }
      return fetchReleaseFromGitHub();
    },
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
    retry: 1,
  });

  // Only offer the CI-published APK. It is signed with the same protected
  // release key as the installed app, so Android can apply it as an update.
  const apkAsset = release ? { browser_download_url: release.apk_url } : undefined;
  const publishedAt = release?.published_at ? new Date(release.published_at) : null;

  return (
    <div className="w-full flex flex-col items-center space-y-6" data-testid="section-download-apk">
      <a
        href={apkAsset?.browser_download_url ?? RELEASES_PAGE_URL}
        download={apkAsset ? 'novacycle-latest.apk' : undefined}
        data-download-url={apkAsset?.browser_download_url}
        data-testid="link-download-apk"
        className="group relative inline-flex items-center justify-center w-full sm:w-auto overflow-hidden rounded-xl bg-primary px-8 py-4 font-medium text-primary-foreground transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background"
      >
        <div className="absolute inset-0 flex h-full w-full justify-center [transform:skew(-12deg)_translateX(-100%)] group-hover:duration-1000 group-hover:[transform:skew(-12deg)_translateX(100%)]">
          <div className="relative h-full w-8 bg-white/20" />
        </div>
        <div className="flex items-center space-x-3">
          <Download className="w-5 h-5" />
          <span className="text-lg font-bold tracking-wide">
            {apkAsset ? 'Download Android APK' : 'Browse APK Releases'}
          </span>
        </div>
      </a>

      <div className="text-center space-y-1">
        {isLoading && (
          <p className="text-sm text-muted-foreground" data-testid="text-release-loading">
            Checking for the newest build…
          </p>
        )}
        {!isLoading && release && (
          <>
            <p className="text-sm text-muted-foreground" data-testid="text-release-version">
              <span className="font-mono text-primary/80">{release.tag_name}</span>
              {publishedAt && (
                <span>
                  {' '}· built{' '}
                  {publishedAt.toLocaleString('en-US', {
                    month: 'short', day: 'numeric', year: 'numeric',
                    hour: 'numeric', minute: '2-digit',
                  })}
                </span>
              )}
            </p>
            {release.stale && (
              <p
                className="text-xs text-muted-foreground/60 flex items-center justify-center space-x-1"
                data-testid="text-release-stale"
              >
                <Clock className="w-3 h-3 shrink-0" />
                <span>Release info may be a few minutes old</span>
              </p>
            )}
          </>
        )}
        {!isLoading && (isError || !release) && (
          <p className="text-sm text-amber-400" data-testid="text-release-unavailable">
            Couldn't fetch the latest release info from GitHub — use the release
            history below to pick the newest build manually.
          </p>
        )}
        <p className="text-xs text-muted-foreground/70 max-w-md">
          New builds publish automatically a few minutes after code is pushed to
          GitHub (push → CI build &amp; tests → release). This page always links the
          newest published build.
        </p>
        <a
          href={RELEASES_PAGE_URL}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-primary/60 hover:text-primary transition-colors inline-flex items-center space-x-1"
        >
          <span>View full release history</span>
          <ExternalLink className="w-3 h-3" />
        </a>
      </div>
    </div>
  );
}

function Dashboard() {
  const [tab, setTab] = useState<'system' | 'performance'>('system');

  return (
    <div className="relative">
      <TrainingStuckBanner />
      <div className="sticky top-0 z-40 w-full border-b border-white/5 bg-background/80 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-6">
          <div className="flex items-center space-x-1 py-2 font-mono text-sm">
            <button
              data-testid="tab-system"
              onClick={() => setTab('system')}
              className={`px-4 py-2 rounded-md tracking-wide transition-colors ${
                tab === 'system'
                  ? 'bg-primary/20 text-primary'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              SYSTEM
            </button>
            <button
              data-testid="tab-performance"
              onClick={() => setTab('performance')}
              className={`px-4 py-2 rounded-md tracking-wide transition-colors ${
                tab === 'performance'
                  ? 'bg-primary/20 text-primary'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              PERFORMANCE
            </button>
          </div>
        </div>
      </div>

      {tab === 'system' ? <StatusDashboard /> : <PerformanceDashboard />}
    </div>
  );
}

function NotFound() {
  return (
    <div className="min-h-[100dvh] flex flex-col items-center justify-center bg-background text-foreground">
      <h1 className="text-4xl font-mono font-bold text-primary mb-4 glow-text-primary">404</h1>
      <p className="text-muted-foreground mb-8 font-mono">Signal lost. Route not found.</p>
      <a href="/" className="text-primary hover:underline font-mono inline-flex items-center space-x-2">
        <Activity className="w-4 h-4" />
        <span>Return to Mission Control</span>
      </a>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
        <Switch>
          <Route path="/" component={Dashboard} />
          <Route component={NotFound} />
        </Switch>
      </WouterRouter>
    </QueryClientProvider>
  );
}

export default App;
