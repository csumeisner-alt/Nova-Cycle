import { useQuery } from '@tanstack/react-query';
import { Star, History } from 'lucide-react';
import { useState } from 'react';

type TierStats = {
  trade_count: number;
  win_rate: number | null;
  avg_return_percent: number | null;
  sufficient_sample: boolean;
};

type TierTrackRecord = {
  ticker: string;
  window: string;
  available_windows: string[];
  overall: TierStats;
  tiers: Record<string, TierStats>;
  excluded_price_data_absent: number;
  min_sample_size: number;
};

const WINDOW_LABELS: Record<string, string> = {
  '30d': '30 DAYS',
  '90d': '90 DAYS',
  all: 'ALL TIME',
};

const TIER_META: Record<string, { label: string; description: string }> = {
  high_conviction: {
    label: 'HIGH-CONVICTION',
    description: 'Signals where multiple checks lined up',
  },
  opportunity: {
    label: 'OPPORTUNITY',
    description: 'Standard signals that passed the basic filters',
  },
};

function fmtPct(v: number | null | undefined): string {
  return typeof v === 'number' ? `${(v * 100).toFixed(0)}%` : '—';
}

function fmtRet(v: number | null | undefined): string {
  if (typeof v !== 'number') return '—';
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
}

function TierRow({
  tierKey,
  stats,
  minSample,
}: {
  tierKey: string;
  stats: TierStats;
  minSample: number;
}) {
  const meta = TIER_META[tierKey];
  const isHigh = tierKey === 'high_conviction';
  return (
    <div
      className={`p-3 rounded-lg border font-mono text-sm ${
        isHigh ? 'border-amber-400/25 bg-amber-400/5' : 'border-white/5 bg-white/[0.02]'
      }`}
      data-testid={`row-tier-${tierKey}`}
    >
      <div className="flex items-center justify-between">
        <span
          className={`inline-flex items-center gap-1 text-xs tracking-wide ${
            isHigh ? 'text-amber-300' : 'text-muted-foreground'
          }`}
        >
          {isHigh && <Star className="w-3 h-3 fill-current" />}
          <span>{meta?.label ?? tierKey.toUpperCase()}</span>
        </span>
        <span className="text-[11px] text-muted-foreground" data-testid={`text-tier-count-${tierKey}`}>
          {stats.trade_count} completed trade{stats.trade_count === 1 ? '' : 's'}
        </span>
      </div>
      {stats.sufficient_sample ? (
        <div className="mt-2 grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-muted-foreground">Win rate</div>
            <div className={`text-lg ${isHigh ? 'text-amber-300' : ''}`} data-testid={`text-tier-winrate-${tierKey}`}>
              {fmtPct(stats.win_rate)}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Avg return / trade</div>
            <div
              className={`text-lg ${
                (stats.avg_return_percent ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}
              data-testid={`text-tier-avgreturn-${tierKey}`}
            >
              {fmtRet(stats.avg_return_percent)}
            </div>
          </div>
        </div>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground leading-snug" data-testid={`text-tier-sparse-${tierKey}`}>
          Not enough {meta ? meta.label.toLowerCase().replace('-', ' ') : tierKey} signals yet — a
          reliable percentage needs at least {minSample} completed trades.
        </p>
      )}
      {meta && <p className="mt-1.5 text-[11px] text-muted-foreground/80">{meta.description}</p>}
    </div>
  );
}

export function TierTrackRecordPanel() {
  const [window, setWindow] = useState<string>('90d');

  const { data, isLoading, isError } = useQuery<TierTrackRecord>({
    queryKey: ['tier-track-record', window],
    queryFn: async () => {
      const res = await fetch(`/api/tier_track_record?ticker=VOO&window=${window}`);
      if (!res.ok) throw new Error(`Track record unavailable (HTTP ${res.status})`);
      return res.json();
    },
    refetchInterval: 5 * 60_000,
    retry: 1,
  });

  const windows = data?.available_windows ?? ['30d', '90d', 'all'];
  const minSample = data?.min_sample_size ?? 5;

  return (
    <div className="mt-8 p-4 bg-white/[0.02] rounded-lg border border-white/5" data-testid="panel-tier-track-record">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center space-x-2 text-muted-foreground">
          <History className="w-4 h-4" />
          <span className="text-sm font-medium tracking-wide">TIER TRACK RECORD</span>
        </div>
        <div className="flex items-center gap-1">
          {windows.map((w) => (
            <button
              key={w}
              onClick={() => setWindow(w)}
              data-testid={`button-window-${w}`}
              className={`text-[10px] font-mono px-2 py-1 rounded border transition-colors ${
                window === w
                  ? 'border-primary/40 bg-primary/10 text-primary'
                  : 'border-white/10 text-muted-foreground hover:bg-white/5'
              }`}
            >
              {WINDOW_LABELS[w] ?? w.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      <p className="text-[11px] text-muted-foreground/80 mb-4 leading-snug" data-testid="text-tier-explainer">
        How signals of each conviction tier actually performed, measured from completed BUY→SELL
        cycles. This is the real history — not a prediction.
      </p>

      {isLoading ? (
        <p className="text-sm text-muted-foreground font-mono">Loading track record…</p>
      ) : isError || !data ? (
        <p className="text-sm text-muted-foreground font-mono" data-testid="text-tier-error">
          Track record unavailable right now.
        </p>
      ) : (
        <div className="space-y-2">
          {(['high_conviction', 'opportunity'] as const).map((k) => (
            <TierRow key={k} tierKey={k} stats={data.tiers[k] ?? { trade_count: 0, win_rate: null, avg_return_percent: null, sufficient_sample: false }} minSample={minSample} />
          ))}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-[11px] font-mono text-muted-foreground">
            <span data-testid="text-tier-overall">
              Overall:{' '}
              {data.overall.sufficient_sample
                ? `${fmtPct(data.overall.win_rate)} win rate · ${fmtRet(data.overall.avg_return_percent)} avg over ${data.overall.trade_count} trades`
                : `${data.overall.trade_count} completed trade${data.overall.trade_count === 1 ? '' : 's'} so far — not enough history yet`}
            </span>
            {data.excluded_price_data_absent > 0 && (
              <span data-testid="text-tier-excluded">
                {data.excluded_price_data_absent} trade{data.excluded_price_data_absent === 1 ? '' : 's'} excluded (missing price data)
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
