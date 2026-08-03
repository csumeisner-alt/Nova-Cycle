import { useQuery } from '@tanstack/react-query';
import { ListOrdered, AlertTriangle } from 'lucide-react';
import { useState } from 'react';

type SignalRow = {
  id: number;
  timestamp: string;
  ticker: string;
  cycle_id: string | null;
  signal_type: 'buy' | 'sell';
  gauge_type: 'long' | 'short';
  confidence: number;
  session_type: string;
  is_extended_hours: boolean;
  conviction_tier: 'opportunity' | 'high_conviction' | null;
  /** "healthy" | "model_unavailable" | "training_stuck" | "stale_rolled_back" | null (pre-tracking rows). */
  model_state: string | null;
};

const WINDOW_LABELS: Record<string, string> = {
  '7d': '7 DAYS',
  '30d': '30 DAYS',
  '90d': '90 DAYS',
};

const MODEL_STATE_LABELS: Record<string, string> = {
  training_stuck: 'training stuck',
  stale_rolled_back: 'stale rollback',
  model_unavailable: 'model unavailable',
};

function isDegraded(state: string | null): boolean {
  return state != null && state !== 'healthy';
}

/** Small red pill flagging a signal stored under a non-healthy model state. */
function DegradedBadge({ state }: { state: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-red-400/30 bg-red-400/10 text-red-400 text-[10px] font-mono tracking-wide"
      title={`Stored while the model was degraded (${MODEL_STATE_LABELS[state] ?? state}) — treat with caution`}
      data-testid="badge-degraded-model"
    >
      <AlertTriangle className="w-3 h-3" />
      DEGRADED
    </span>
  );
}

export function SignalHistoryPanel() {
  const [window, setWindow] = useState<string>('30d');

  const { data, isLoading, isError } = useQuery<SignalRow[]>({
    queryKey: ['signal-history', window],
    queryFn: async () => {
      const res = await fetch(`/api/signal_history?ticker=VOO&window=${window}`);
      if (!res.ok) throw new Error(`Signal history unavailable (HTTP ${res.status})`);
      return res.json();
    },
    refetchInterval: 5 * 60_000,
    retry: 1,
  });

  // Newest first for review.
  const rows = (data ?? []).slice().reverse();
  const degradedCount = rows.filter((r) => isDegraded(r.model_state)).length;

  return (
    <div className="mt-8 p-4 bg-white/[0.02] rounded-lg border border-white/5" data-testid="panel-signal-history">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center space-x-2 text-muted-foreground">
          <ListOrdered className="w-4 h-4" />
          <span className="text-sm font-medium tracking-wide">SIGNAL HISTORY</span>
        </div>
        <div className="flex items-center gap-1">
          {Object.keys(WINDOW_LABELS).map((w) => (
            <button
              key={w}
              onClick={() => setWindow(w)}
              data-testid={`button-signal-window-${w}`}
              className={`text-[10px] font-mono px-2 py-1 rounded border transition-colors ${
                window === w
                  ? 'border-primary/40 bg-primary/10 text-primary'
                  : 'border-white/10 text-muted-foreground hover:bg-white/5'
              }`}
            >
              {WINDOW_LABELS[w]}
            </button>
          ))}
        </div>
      </div>
      <p className="text-[11px] text-muted-foreground/80 mb-4 leading-snug" data-testid="text-signal-history-explainer">
        Every stored BUY/SELL signal. Signals recorded while a model was degraded are flagged —
        they slipped through before the notification gate and should be read with caution.
        {degradedCount > 0 && (
          <span className="text-red-400"> {degradedCount} degraded in this window.</span>
        )}
      </p>

      {isLoading ? (
        <p className="text-sm text-muted-foreground font-mono">Loading signal history…</p>
      ) : isError ? (
        <p className="text-sm text-muted-foreground font-mono" data-testid="text-signal-history-error">
          Signal history unavailable right now.
        </p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-muted-foreground font-mono" data-testid="text-signal-history-empty">
          No signals recorded in this window.
        </p>
      ) : (
        <div className="space-y-1 max-h-80 overflow-y-auto pr-1">
          {rows.map((r) => {
            const degraded = isDegraded(r.model_state);
            const isBuy = r.signal_type === 'buy';
            return (
              <div
                key={r.id}
                className={`flex flex-wrap items-center gap-2 px-3 py-2 rounded border font-mono text-xs ${
                  degraded ? 'border-red-400/25 bg-red-400/5' : 'border-white/5 bg-white/[0.02]'
                }`}
                data-testid={`row-signal-${r.id}`}
              >
                <span className={`font-bold ${isBuy ? 'text-emerald-400' : 'text-red-400'}`}>
                  {r.signal_type.toUpperCase()}
                </span>
                <span className="text-muted-foreground">{r.gauge_type}</span>
                <span>{(r.confidence * 100).toFixed(0)}%</span>
                {r.conviction_tier === 'high_conviction' && (
                  <span className="text-amber-300 text-[10px]">★ HIGH-CONVICTION</span>
                )}
                {degraded && <DegradedBadge state={r.model_state as string} />}
                <span className="ml-auto text-muted-foreground/70">
                  {new Date(r.timestamp).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: false,
                  })}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
