import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { Activity, Server, Clock, Download, ExternalLink, Terminal, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import { useState, useEffect } from 'react';

const queryClient = new QueryClient();

// Add a scanline effect via a fixed overlay
function Scanlines() {
  return (
    <div className="pointer-events-none fixed inset-0 z-50 opacity-[0.03] scanline mix-blend-overlay" />
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