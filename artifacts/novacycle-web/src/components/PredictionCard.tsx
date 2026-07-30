import { useQuery } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { confidenceZone } from '@/lib/confidenceZone';
import { useState } from 'react';

export type PredictionDisplay = {
  confidence_percent?: number;
  trend?: 'UP' | 'DOWN' | 'NEUTRAL';
  display_signal?: 'BUY BIAS' | 'SELL BIAS' | 'NEUTRAL / HOLD';
  note?: string;
  data_quality_degraded?: boolean;
  data_quality_reason?: string;
};

function TrendArrow({ trend }: { trend: PredictionDisplay['trend'] }) {
  if (trend === 'UP') return <TrendingUp className="w-4 h-4 text-emerald-400" data-testid="icon-trend-up" />;
  if (trend === 'DOWN') return <TrendingDown className="w-4 h-4 text-red-400" data-testid="icon-trend-down" />;
  return <Minus className="w-4 h-4 text-muted-foreground" data-testid="icon-trend-neutral" />;
}

export function PredictionCard({ name, label }: { name: string; label: string }) {
  const [showReason, setShowReason] = useState(false);

  const { data, isLoading, isError } = useQuery<PredictionDisplay>({
    queryKey: ['predict', name],
    queryFn: async () => {
      const res = await fetch(`/api/predict_${name}?ticker=VOO`, { method: 'POST' });
      if (!res.ok) throw new Error(`Prediction unavailable (HTTP ${res.status})`);
      return res.json();
    },
    refetchInterval: 60000,
    retry: 1,
  });

  const pct =
    typeof data?.confidence_percent === 'number'
      ? Math.max(0, Math.min(100, data.confidence_percent))
      : null;
  const trend = data?.trend;
  const signal = data?.display_signal;
  const noData = isError || !data || pct === null || trend === undefined || signal === undefined;

  if (isLoading) {
    return (
      <div className="p-4 bg-black/30 rounded-lg border border-white/5 space-y-3" data-testid={`prediction-card-${name}`}>
        <div className="text-xs tracking-wide uppercase font-mono">{label}</div>
        <div className="h-8 w-24 bg-white/5 rounded animate-pulse" />
        <div className="h-2 w-full bg-white/5 rounded animate-pulse" />
      </div>
    );
  }

  if (noData) {
    return (
      <div
        className="p-4 bg-black/30 rounded-lg border border-white/5 space-y-3"
        data-testid={`prediction-card-${name}`}
      >
        <div className="flex items-center justify-between">
          <span className="text-xs tracking-wide uppercase font-mono">{label}</span>
          <span
            className="inline-flex items-center space-x-1 text-muted-foreground bg-white/5 border border-white/10 px-2 py-0.5 rounded-full text-xs font-mono"
            data-testid={`badge-signal-${name}`}
          >
            <Minus className="w-3 h-3" />
            <span>NO DATA</span>
          </span>
        </div>
        <div className="font-mono text-2xl text-muted-foreground" data-testid={`text-confidence-${name}`}>
          —
        </div>
        <div className="h-2 w-full rounded-full bg-white/5" />
        <p className="text-[11px] text-muted-foreground font-mono">
          Prediction unavailable — backend offline or no historical data yet.
        </p>
      </div>
    );
  }

  const zone = confidenceZone(pct);
  const degraded = data?.data_quality_degraded === true;
  const degradedReason = data?.data_quality_reason ?? '';

  return (
    <div className="p-4 bg-black/30 rounded-lg border border-white/5 space-y-3" data-testid={`prediction-card-${name}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs tracking-wide uppercase font-mono">{label}</span>
        <span
          className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-xs font-mono border ${zone.text} ${zone.bg} ${zone.border}`}
          data-testid={`badge-signal-${name}`}
        >
          <TrendArrow trend={trend} />
          <span>{signal}</span>
        </span>
      </div>
      <div className="flex items-baseline space-x-2">
        <span className={`font-mono text-2xl ${zone.text}`} data-testid={`text-confidence-${name}`}>
          {pct}%
        </span>
        <span className="text-xs text-muted-foreground font-mono">confidence</span>
        <span className="ml-auto text-xs font-mono text-muted-foreground" data-testid={`text-trend-${name}`}>
          TREND: {trend}
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-white/5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${zone.bar}`}
          style={{ width: `${pct}%` }}
          data-testid={`bar-confidence-${name}`}
        />
      </div>
      {degraded && (
        <div
          className="rounded-md border border-amber-400/30 bg-amber-400/5 px-3 py-2 space-y-1"
          data-testid={`banner-data-quality-${name}`}
        >
          <button
            type="button"
            onClick={() => setShowReason((v) => !v)}
            className="flex w-full items-center justify-between text-amber-400 gap-2"
            aria-expanded={showReason}
            aria-label="Toggle data quality detail"
          >
            <span className="flex items-center gap-1.5 text-xs font-mono font-medium">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
              ⚠ Data quality issue: one or more candles were filtered. Signal may be less reliable.
            </span>
            {showReason ? (
              <ChevronUp className="w-3.5 h-3.5 shrink-0" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5 shrink-0" />
            )}
          </button>
          {showReason && degradedReason && (
            <p
              className="text-[11px] font-mono text-amber-200/70 leading-relaxed break-words"
              data-testid={`text-data-quality-reason-${name}`}
            >
              {degradedReason}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
