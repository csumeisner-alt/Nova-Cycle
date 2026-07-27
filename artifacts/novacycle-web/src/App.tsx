import { QueryClient, QueryClientProvider, useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Activity, Server, Clock, Download, ExternalLink, Terminal, AlertTriangle, CheckCircle2, RotateCcw, KeyRound, X } from 'lucide-react';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import { useState, useEffect } from 'react';

const queryClient = new QueryClient();

// Add a scanline effect via a fixed overlay
function Scanlines() {
  return (
    <div className="pointer-events-none fixed inset-0 z-50 opacity-[0.03] scanline mix-blend-overlay" />
  );
}

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
            
            {!isLoading && !isError && health && <RetrainStatusPanel health={health} />}

            {!isLoading && !isError && health && <FallbackHistoryPanel health={health} />}

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
        <div className="w-full flex flex-col items-center space-y-6">
          <a
            href="https://github.com/csumeisner-alt/Nova-Cycle/releases/download/latest/app-release.apk"
            className="group relative inline-flex items-center justify-center w-full sm:w-auto overflow-hidden rounded-xl bg-primary px-8 py-4 font-medium text-primary-foreground transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background"
          >
            <div className="absolute inset-0 flex h-full w-full justify-center [transform:skew(-12deg)_translateX(-100%)] group-hover:duration-1000 group-hover:[transform:skew(-12deg)_translateX(100%)]">
              <div className="relative h-full w-8 bg-white/20" />
            </div>
            <div className="flex items-center space-x-3">
              <Download className="w-5 h-5" />
              <span className="text-lg font-bold tracking-wide">Download Android APK</span>
            </div>
          </a>
          
          <div className="text-center space-y-1">
            <p className="text-sm text-muted-foreground">
              v1.0.0 (Latest Release)
            </p>
            <a 
              href="https://github.com/csumeisner-alt/Nova-Cycle/releases" 
              target="_blank" 
              rel="noreferrer"
              className="text-xs text-primary/60 hover:text-primary transition-colors inline-flex items-center space-x-1"
            >
              <span>View full release history</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>

      </main>
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
          <Route path="/" component={StatusDashboard} />
          <Route component={NotFound} />
        </Switch>
      </WouterRouter>
    </QueryClientProvider>
  );
}

export default App;