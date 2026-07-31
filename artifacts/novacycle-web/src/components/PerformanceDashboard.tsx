import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  LabelList,
  Cell,
  Label,
} from 'recharts';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from '@/components/ui/chart';
import {
  BarChart3,
  Target,
  TrendingUp,
  AlertTriangle,
  Flame,
  Repeat,
  ArrowUpRight,
  ArrowDownRight,
  ArrowUpDown,
} from 'lucide-react';

// ─── palette ──────────────────────────────────────────────────────────────
const GREEN = '#22c55e';
const RED = '#ef4444';
const BLUE = '#3b82f6';
const MODEL_COLORS = [BLUE, GREEN, '#a855f7', '#f59e0b', '#ec4899', '#14b8a6'];

// ─── types (matches GET /api/model_performance contract) ────────────────────
type Summary = {
  total_trades: number;
  wins: number;
  losses: number;
  buy_precision: number;
  avg_return_percent: number;
  missed_rally_rate: number;
  current_win_streak: number;
  recommendation_stability: number;
  avg_confidence: number;
  cumulative_return_percent: number;
};

type PeriodRow = {
  label: string;
  start: string;
  buy_count: number;
  wins: number;
  losses: number;
  precision: number;
  avg_return_percent: number;
  missed_rallies: number;
  avg_confidence: number;
  oos_accuracy: number | null;
};

type BucketStat = { trade_count: number; win_rate: number; avg_return_percent: number };
type CalibrationPoint = { confidence_mid: number; actual_win_rate: number | null; trade_count: number };
type PnlPoint = { timestamp: string; cumulative_return_percent: number };
type DistBin = { label: string; min: number; max: number; count: number };
type SliceStat = { count: number; win_rate: number; average_return_percent: number };
type Trade = {
  cycle_id?: string | number;
  buy_timestamp: string;
  sell_timestamp: string;
  return_percent: number;
  hold_time_minutes: number;
  confidence_at_buy?: number;
} | null;
type AccuracyPoint = { model_name: string; trained_at: string; accuracy: number | null };

type PerformanceResponse = {
  ticker: string;
  period: string;
  window: string;
  summary: Summary;
  periods: PeriodRow[];
  confidence_buckets: { low: BucketStat; medium: BucketStat; high: BucketStat };
  calibration_curve: CalibrationPoint[];
  cumulative_pnl: PnlPoint[];
  return_distribution: DistBin[];
  session_breakdown: Record<string, SliceStat>;
  vix_regime_breakdown: Record<string, SliceStat>;
  best_trade: Trade;
  worst_trade: Trade;
  streak: { current_win: number; current_loss: number; longest_win: number; longest_loss: number };
  missed_rallies: { count: number; timestamps: string[]; rate: number };
  accuracy_history: AccuracyPoint[];
};

// ─── formatting helpers ─────────────────────────────────────────────────────
const fmtPct = (v: number | null | undefined): string =>
  typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(1)}%` : '—';
// Fractions (0-1) rendered as percentages
const fmtFrac = (v: number | null | undefined): string =>
  typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—';
const fmtDollar = (v: number | null | undefined): string =>
  typeof v === 'number' && Number.isFinite(v) ? `$${v.toFixed(2)}` : '—';
const fmtInt = (v: number | null | undefined): string =>
  typeof v === 'number' && Number.isFinite(v) ? `${Math.round(v)}` : '—';
const fmtDate = (iso: string | null | undefined): string => {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isFinite(d.getTime())
    ? d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    : '—';
};
const fmtHold = (mins: number | null | undefined): string => {
  if (typeof mins !== 'number' || !Number.isFinite(mins)) return '—';
  if (mins < 60) return `${Math.round(mins)}m`;
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  return m ? `${h}h ${m}m` : `${h}h`;
};

// ─── period / confidence filters ────────────────────────────────────────────
type Period = 'day' | 'week' | 'month';
type ConfKey = 'all' | 'low' | 'medium' | 'high';
const CONF_RANGES: Record<ConfKey, { min?: number; max?: number; label: string }> = {
  all: { label: 'All' },
  low: { min: 0.0, max: 0.4, label: 'Low 0–40%' },
  medium: { min: 0.4, max: 0.7, label: 'Medium 40–70%' },
  high: { min: 0.7, max: 1.0, label: 'High 70–100%' },
};

const EMPTY_MSG =
  'No completed trades yet. This fills in automatically as BUY→SELL cycles close. Check back after the first SELL signal.';
const EMPTY_CONF_MSG = 'No trades recorded at this confidence level yet.';

// ─── small presentational pieces ────────────────────────────────────────────
function Panel({
  title,
  icon,
  testId,
  children,
  className = '',
}: {
  title: string;
  icon?: React.ReactNode;
  testId?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`p-4 bg-white/[0.02] rounded-lg border border-white/5 ${className}`}
      data-testid={testId}
    >
      <div className="flex items-center space-x-2 text-muted-foreground mb-4">
        {icon}
        <span className="text-sm font-medium tracking-wide">{title}</span>
      </div>
      {children}
    </div>
  );
}

function EmptyState({ testId, message }: { testId: string; message: string }) {
  return (
    <div
      className="flex flex-col items-center justify-center text-center h-64 rounded-md border border-dashed border-white/10 bg-black/20 px-6"
      data-testid={testId}
    >
      <BarChart3 className="w-6 h-6 text-muted-foreground/50 mb-3" />
      <p className="text-sm text-muted-foreground font-mono max-w-sm leading-relaxed">{message}</p>
    </div>
  );
}

function SkeletonBlock({ className = '' }: { className?: string }) {
  return <div className={`bg-white/5 rounded animate-pulse ${className}`} />;
}

function PerformanceSkeleton() {
  return (
    <div className="space-y-6" data-testid="skeleton-performance">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <SkeletonBlock key={i} className="h-24" />
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <SkeletonBlock className="h-12" />
        <SkeletonBlock className="h-12" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonBlock key={i} className="h-72" />
        ))}
      </div>
      <SkeletonBlock className="h-72" />
      <SkeletonBlock className="h-64" />
    </div>
  );
}

// ─── summary card ───────────────────────────────────────────────────────────
function SummaryCard({
  name,
  title,
  value,
  icon,
  hint,
}: {
  name: string;
  title: string;
  value: string;
  icon: React.ReactNode;
  hint?: string;
}) {
  return (
    <div
      className="p-4 bg-white/[0.02] rounded-lg border border-white/5 space-y-2"
      data-testid={`card-${name}`}
    >
      <div className="flex items-center space-x-2 text-muted-foreground">
        {icon}
        <span className="text-[11px] font-mono uppercase tracking-wide">{title}</span>
      </div>
      <div className="font-mono text-2xl text-foreground">{value}</div>
      {hint && <div className="text-[10px] text-muted-foreground/70 font-mono">{hint}</div>}
    </div>
  );
}

// ─── trade callout row ──────────────────────────────────────────────────────
function TradeCallout({ label, trade, kind }: { label: string; trade: Trade; kind: 'best' | 'worst' }) {
  const positive = kind === 'best';
  const colorText = positive ? 'text-emerald-400' : 'text-red-400';
  const colorBorder = positive ? 'border-emerald-400/20' : 'border-red-400/20';
  const colorBg = positive ? 'bg-emerald-400/10' : 'bg-red-400/10';
  return (
    <div
      className={`p-3 rounded-lg border font-mono text-sm ${colorBorder} ${colorBg}`}
      data-testid={`callout-${kind}-trade`}
    >
      <div className="flex items-center justify-between">
        <span className={`flex items-center space-x-1 text-xs uppercase tracking-wide ${colorText}`}>
          {positive ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
          <span>{label}</span>
        </span>
        {trade ? (
          <span className={`text-lg ${colorText}`}>{fmtPct(trade.return_percent)}</span>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </div>
      {trade ? (
        <div className="mt-1.5 flex flex-col sm:flex-row sm:items-center sm:space-x-4 space-y-0.5 sm:space-y-0 text-[11px] text-muted-foreground">
          <span>Closed: {fmtDate(trade.sell_timestamp)}</span>
          <span>Hold: {fmtHold(trade.hold_time_minutes)}</span>
        </div>
      ) : (
        <div className="mt-1.5 text-[11px] text-muted-foreground">No trades recorded yet.</div>
      )}
    </div>
  );
}

// ─── sortable periods table ─────────────────────────────────────────────────
type SortKey =
  | 'label'
  | 'buy_count'
  | 'wins'
  | 'losses'
  | 'precision'
  | 'avg_return_percent'
  | 'missed_rallies'
  | 'avg_confidence';

const COLUMNS: { key: SortKey; header: string }[] = [
  { key: 'label', header: 'Period' },
  { key: 'buy_count', header: 'BUY' },
  { key: 'wins', header: 'Wins' },
  { key: 'losses', header: 'Losses' },
  { key: 'precision', header: 'Precision' },
  { key: 'avg_return_percent', header: 'Avg Return' },
  { key: 'missed_rallies', header: 'Missed' },
  { key: 'avg_confidence', header: 'Avg Conf.' },
];

function PeriodsTable({ rows }: { rows: PeriodRow[] }) {
  const [sortKey, setSortKey] = useState<SortKey>('label');
  const [asc, setAsc] = useState(true);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      let cmp: number;
      if (typeof av === 'number' && typeof bv === 'number') cmp = av - bv;
      else cmp = String(av).localeCompare(String(bv));
      return asc ? cmp : -cmp;
    });
    return copy;
  }, [rows, sortKey, asc]);

  const toggle = (key: SortKey) => {
    if (key === sortKey) setAsc((v) => !v);
    else {
      setSortKey(key);
      setAsc(true);
    }
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left font-mono text-xs" data-testid="table-periods">
        <thead>
          <tr className="text-muted-foreground border-b border-white/10">
            {COLUMNS.map((c) => (
              <th
                key={c.key}
                className="py-2 px-2 cursor-pointer select-none hover:text-foreground transition-colors"
                data-testid={`th-${c.key}`}
                onClick={() => toggle(c.key)}
              >
                <span className="inline-flex items-center space-x-1">
                  <span>{c.header}</span>
                  {sortKey === c.key ? (
                    <span className="text-primary">{asc ? '▲' : '▼'}</span>
                  ) : (
                    <ArrowUpDown className="w-3 h-3 opacity-40" />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.label} className="border-b border-white/5 hover:bg-white/[0.02]" data-testid={`row-period-${r.label}`}>
              <td className="py-2 px-2 text-foreground">{r.label}</td>
              <td className="py-2 px-2">{fmtInt(r.buy_count)}</td>
              <td className="py-2 px-2 text-emerald-400">{fmtInt(r.wins)}</td>
              <td className="py-2 px-2 text-red-400">{fmtInt(r.losses)}</td>
              <td className="py-2 px-2">{fmtFrac(r.precision)}</td>
              <td className={`py-2 px-2 ${r.avg_return_percent >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {fmtPct(r.avg_return_percent)}
              </td>
              <td className="py-2 px-2">{fmtInt(r.missed_rallies)}</td>
              <td className="py-2 px-2">{fmtFrac(r.avg_confidence)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── slice breakdown cards ──────────────────────────────────────────────────
function BreakdownCards({
  title,
  icon,
  testId,
  data,
}: {
  title: string;
  icon: React.ReactNode;
  testId: string;
  data: Record<string, SliceStat>;
}) {
  const entries = Object.entries(data ?? {});
  return (
    <Panel title={title} icon={icon} testId={testId}>
      {entries.length === 0 ? (
        <p className="text-xs text-muted-foreground font-mono">No data yet.</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {entries.map(([name, s]) => (
            <div
              key={name}
              className="p-3 bg-black/30 rounded-lg border border-white/5 space-y-1 font-mono"
              data-testid={`slice-${testId}-${name}`}
            >
              <div className="text-[11px] uppercase tracking-wide text-muted-foreground truncate">{name}</div>
              <div className="text-xs">
                Win rate: <span className="text-foreground">{fmtFrac(s.win_rate)}</span>
              </div>
              <div className="text-xs">
                Avg return:{' '}
                <span className={s.average_return_percent >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                  {fmtPct(s.average_return_percent)}
                </span>
              </div>
              <div className="text-[10px] text-muted-foreground/70">{fmtInt(s.count)} trades</div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

// ─── chart configs ──────────────────────────────────────────────────────────
const pnlConfig: ChartConfig = {
  cumulative_return_percent: { label: 'Cumulative Return', color: BLUE },
};
const winLossConfig: ChartConfig = {
  wins: { label: 'Wins', color: GREEN },
  losses: { label: 'Losses', color: RED },
};
const calibrationConfig: ChartConfig = {
  actual_win_rate: { label: 'Actual Win Rate', color: BLUE },
};
const distConfig: ChartConfig = {
  count: { label: 'Trades', color: BLUE },
};

// ─── main component ─────────────────────────────────────────────────────────
export function PerformanceDashboard() {
  const [period, setPeriod] = useState<Period>('day');
  const [conf, setConf] = useState<ConfKey>('all');

  const range = CONF_RANGES[conf];

  const { data, isLoading, isError, error } = useQuery<PerformanceResponse>({
    queryKey: ['model_performance', period, conf],
    queryFn: async () => {
      const params = new URLSearchParams({ ticker: 'VOO', period });
      if (range.min !== undefined) params.set('confidence_min', String(range.min));
      if (range.max !== undefined) params.set('confidence_max', String(range.max));
      const res = await fetch(`/api/model_performance?${params.toString()}`);
      if (!res.ok) throw new Error(`Performance data unavailable (HTTP ${res.status})`);
      return res.json();
    },
    refetchInterval: 60000,
    retry: 1,
  });

  const periods: Period[] = ['day', 'week', 'month'];
  const confKeys: ConfKey[] = ['all', 'low', 'medium', 'high'];

  const hasTrades = (data?.summary?.total_trades ?? 0) > 0;
  const isFiltered = conf !== 'all';

  // Accuracy history transformed to one line per model
  const accuracyData = useMemo(() => {
    // Backend may emit accuracy: null for retrains recorded without a metric —
    // skip those points so the chart never plots bogus values.
    const history = (data?.accuracy_history ?? []).filter(
      (h) => typeof h.accuracy === 'number' && Number.isFinite(h.accuracy),
    );
    const models = Array.from(new Set(history.map((h) => h.model_name)));
    const byTs = new Map<string, Record<string, number | string>>();
    for (const h of history) {
      const key = h.trained_at;
      const existing = byTs.get(key) ?? { trained_at: key };
      existing[h.model_name] = (h.accuracy as number) * 100;
      byTs.set(key, existing);
    }
    const rows = Array.from(byTs.values()).sort((a, b) =>
      String(a.trained_at).localeCompare(String(b.trained_at)),
    );
    return { models, rows };
  }, [data?.accuracy_history]);

  const calibrationData = useMemo(
    () =>
      (data?.calibration_curve ?? []).map((c) => ({
        x: c.confidence_mid * 100,
        y: c.actual_win_rate === null ? null : c.actual_win_rate * 100,
        trade_count: c.trade_count,
      })),
    [data?.calibration_curve],
  );

  const pnlData = useMemo(
    () =>
      (data?.cumulative_pnl ?? []).map((p) => ({
        timestamp: p.timestamp,
        cumulative_return_percent: p.cumulative_return_percent,
      })),
    [data?.cumulative_pnl],
  );

  const winLossData = useMemo(
    () =>
      (data?.periods ?? []).map((p) => ({
        label: p.label,
        wins: p.wins,
        losses: p.losses,
      })),
    [data?.periods],
  );

  const summary = data?.summary;

  // ── header controls ──
  const controls = (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
      <div className="inline-flex rounded-lg border border-white/10 bg-black/30 p-1 font-mono text-xs">
        {periods.map((p) => (
          <button
            key={p}
            data-testid={`toggle-period-${p}`}
            onClick={() => setPeriod(p)}
            className={`px-3 py-1.5 rounded-md transition-colors uppercase tracking-wide ${
              period === p ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {p}
          </button>
        ))}
      </div>
      <div className="inline-flex flex-wrap rounded-lg border border-white/10 bg-black/30 p-1 font-mono text-xs gap-1">
        {confKeys.map((c) => (
          <button
            key={c}
            data-testid={`filter-confidence-${c}`}
            onClick={() => setConf(c)}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              conf === c ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {CONF_RANGES[c].label}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-[100dvh] w-full relative overflow-hidden bg-background">
      <main className="relative z-10 w-full max-w-7xl mx-auto px-6 py-8">
        {controls}

        {isLoading ? (
          <PerformanceSkeleton />
        ) : isError ? (
          <div
            className="p-4 bg-destructive/5 rounded-lg border border-destructive/10 font-mono text-xs text-destructive"
            data-testid="error-performance"
          >
            Failed to load performance data.
            <span className="text-muted-foreground mt-2 block">
              {error instanceof Error ? error.message : 'Unknown error'}
            </span>
          </div>
        ) : !hasTrades && isFiltered ? (
          <EmptyState testId="empty-confidence" message={EMPTY_CONF_MSG} />
        ) : !hasTrades ? (
          <EmptyState testId="empty-performance" message={EMPTY_MSG} />
        ) : (
          <div className="space-y-6">
            {/* summary cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              <SummaryCard
                name="buy-precision"
                title="BUY Precision"
                value={fmtFrac(summary?.buy_precision)}
                icon={<Target className="w-4 h-4" />}
              />
              <SummaryCard
                name="avg-return"
                title="Avg Return / Trade"
                value={fmtPct(summary?.avg_return_percent)}
                icon={<TrendingUp className="w-4 h-4" />}
              />
              <SummaryCard
                name="missed-rally-rate"
                title="Missed Rally Rate"
                value={fmtFrac(summary?.missed_rally_rate)}
                icon={<AlertTriangle className="w-4 h-4" />}
              />
              <SummaryCard
                name="current-win-streak"
                title="Current Win Streak"
                value={fmtInt(summary?.current_win_streak)}
                icon={<Flame className="w-4 h-4" />}
              />
              <SummaryCard
                name="recommendation-stability"
                title="Rec. Stability"
                value={fmtInt(summary?.recommendation_stability)}
                hint="avg signal flips / day"
                icon={<Repeat className="w-4 h-4" />}
              />
            </div>

            {/* best / worst callouts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <TradeCallout label="Best Trade" trade={data?.best_trade ?? null} kind="best" />
              <TradeCallout label="Worst Trade" trade={data?.worst_trade ?? null} kind="worst" />
            </div>

            {/* charts grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* cumulative pnl */}
              <Panel title="CUMULATIVE P&L" icon={<TrendingUp className="w-4 h-4" />}>
                <div className="-mt-2 mb-3 font-mono text-xs text-muted-foreground" data-testid="text-cumulative-pnl">
                  Net:{' '}
                  <span className={(summary?.cumulative_return_percent ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                    {fmtPct(summary?.cumulative_return_percent)}
                  </span>{' '}
                  <span className="text-muted-foreground/60">
                    ({fmtDollar(summary?.cumulative_return_percent)} per $100)
                  </span>
                </div>
                {pnlData.length === 0 ? (
                  <EmptyState testId="empty-cumulative-pnl" message={EMPTY_MSG} />
                ) : (
                  <ChartContainer config={pnlConfig} className="h-64 w-full aspect-auto" data-testid="chart-cumulative-pnl">
                    <LineChart data={pnlData} margin={{ top: 10, right: 12, left: 4, bottom: 24 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        dataKey="timestamp"
                        tickFormatter={(v) => fmtDate(v)}
                        tick={{ fontSize: 10 }}
                        minTickGap={24}
                      >
                        <Label value="Time" position="insideBottom" offset={-12} fontSize={11} fill="currentColor" />
                      </XAxis>
                      <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10 }} width={48}>
                        <Label value="Return (%)" angle={-90} position="insideLeft" fontSize={11} fill="currentColor" />
                      </YAxis>
                      <ReferenceLine y={0} stroke="rgba(255,255,255,0.35)" strokeDasharray="4 4" />
                      <ChartTooltip
                        content={
                          <ChartTooltipContent
                            labelFormatter={(v) => fmtDate(v as string)}
                            formatter={(value) => [`${fmtPct(value as number)}`, ' Cumulative Return']}
                          />
                        }
                      />
                      <Line
                        type="monotone"
                        dataKey="cumulative_return_percent"
                        stroke={BLUE}
                        strokeWidth={2}
                        dot={false}
                        name="Cumulative Return"
                      />
                    </LineChart>
                  </ChartContainer>
                )}
              </Panel>

              {/* wins vs losses */}
              <Panel title="WINS VS LOSSES" icon={<BarChart3 className="w-4 h-4" />}>
                {winLossData.length === 0 ? (
                  <EmptyState testId="empty-wins-losses" message={EMPTY_MSG} />
                ) : (
                  <ChartContainer config={winLossConfig} className="h-64 w-full aspect-auto" data-testid="chart-wins-losses">
                    <BarChart data={winLossData} margin={{ top: 16, right: 12, left: 4, bottom: 24 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                      <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={12}>
                        <Label value="Period" position="insideBottom" offset={-12} fontSize={11} fill="currentColor" />
                      </XAxis>
                      <YAxis allowDecimals={false} tick={{ fontSize: 10 }} width={40}>
                        <Label value="Trades" angle={-90} position="insideLeft" fontSize={11} fill="currentColor" />
                      </YAxis>
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <ChartLegend content={<ChartLegendContent />} />
                      <Bar dataKey="wins" fill={GREEN} name="Wins" radius={[2, 2, 0, 0]}>
                        <LabelList dataKey="wins" position="top" fontSize={10} fill="#e5e7eb" formatter={(v: number) => fmtInt(v)} />
                      </Bar>
                      <Bar dataKey="losses" fill={RED} name="Losses" radius={[2, 2, 0, 0]}>
                        <LabelList dataKey="losses" position="top" fontSize={10} fill="#e5e7eb" formatter={(v: number) => fmtInt(v)} />
                      </Bar>
                    </BarChart>
                  </ChartContainer>
                )}
              </Panel>

              {/* calibration */}
              <Panel title="CONFIDENCE CALIBRATION" icon={<Target className="w-4 h-4" />}>
                {calibrationData.length === 0 ? (
                  <EmptyState testId="empty-calibration" message={EMPTY_MSG} />
                ) : (
                  <ChartContainer config={calibrationConfig} className="h-64 w-full aspect-auto" data-testid="chart-calibration">
                    <ScatterChart margin={{ top: 10, right: 12, left: 4, bottom: 24 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        type="number"
                        dataKey="x"
                        domain={[0, 100]}
                        tickFormatter={(v) => `${v}%`}
                        tick={{ fontSize: 10 }}
                      >
                        <Label value="Confidence (%)" position="insideBottom" offset={-12} fontSize={11} fill="currentColor" />
                      </XAxis>
                      <YAxis
                        type="number"
                        dataKey="y"
                        domain={[0, 100]}
                        tickFormatter={(v) => `${v}%`}
                        tick={{ fontSize: 10 }}
                        width={48}
                      >
                        <Label value="Actual Win Rate (%)" angle={-90} position="insideLeft" fontSize={11} fill="currentColor" />
                      </YAxis>
                      <ChartTooltip
                        content={
                          <ChartTooltipContent
                            labelFormatter={() => 'Calibration'}
                            formatter={(value, name) => [
                              `${fmtPct(value as number)}`,
                              name === 'y' ? ' Actual Win Rate' : ` ${name}`,
                            ]}
                          />
                        }
                      />
                      {/* perfect calibration diagonal */}
                      <ReferenceLine
                        segment={[
                          { x: 0, y: 0 },
                          { x: 100, y: 100 },
                        ]}
                        stroke="rgba(255,255,255,0.35)"
                        strokeDasharray="6 4"
                        data-testid="calibration-reference-line"
                        ifOverflow="extendDomain"
                        label={{ value: 'Perfect calibration', position: 'insideTopRight', fontSize: 9, fill: 'rgba(255,255,255,0.5)' }}
                      />
                      <Scatter name="Actual Win Rate" data={calibrationData} fill={BLUE} line={{ stroke: BLUE }} />
                    </ScatterChart>
                  </ChartContainer>
                )}
              </Panel>

              {/* accuracy over time */}
              <Panel title="MODEL ACCURACY OVER TIME" icon={<TrendingUp className="w-4 h-4" />}>
                {accuracyData.rows.length === 0 ? (
                  <EmptyState testId="empty-accuracy" message={EMPTY_MSG} />
                ) : (
                  <ChartContainer
                    config={Object.fromEntries(
                      accuracyData.models.map((m, i) => [m, { label: m, color: MODEL_COLORS[i % MODEL_COLORS.length] }]),
                    )}
                    className="h-64 w-full aspect-auto"
                    data-testid="chart-accuracy"
                  >
                    <LineChart data={accuracyData.rows} margin={{ top: 10, right: 12, left: 4, bottom: 24 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        dataKey="trained_at"
                        tickFormatter={(v) => fmtDate(v)}
                        tick={{ fontSize: 10 }}
                        minTickGap={24}
                      >
                        <Label value="Trained At" position="insideBottom" offset={-12} fontSize={11} fill="currentColor" />
                      </XAxis>
                      <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10 }} width={48}>
                        <Label value="Accuracy (%)" angle={-90} position="insideLeft" fontSize={11} fill="currentColor" />
                      </YAxis>
                      <ChartTooltip
                        content={
                          <ChartTooltipContent
                            labelFormatter={(v) => fmtDate(v as string)}
                            formatter={(value, name) => [`${fmtPct(value as number)}`, ` ${name}`]}
                          />
                        }
                      />
                      <ChartLegend content={<ChartLegendContent />} />
                      {accuracyData.models.map((m, i) => (
                        <Line
                          key={m}
                          type="monotone"
                          dataKey={m}
                          name={m}
                          stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                          strokeWidth={2}
                          dot={{ r: 2 }}
                          connectNulls
                        />
                      ))}
                    </LineChart>
                  </ChartContainer>
                )}
              </Panel>
            </div>

            {/* return distribution (full width) */}
            <Panel title="RETURN DISTRIBUTION" icon={<BarChart3 className="w-4 h-4" />}>
              {(data?.return_distribution?.length ?? 0) === 0 ? (
                <EmptyState testId="empty-return-distribution" message={EMPTY_MSG} />
              ) : (
                <ChartContainer config={distConfig} className="h-64 w-full aspect-auto" data-testid="chart-return-distribution">
                  <BarChart data={data?.return_distribution ?? []} margin={{ top: 16, right: 12, left: 4, bottom: 24 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={4}>
                      <Label value="Return (%)" position="insideBottom" offset={-12} fontSize={11} fill="currentColor" />
                    </XAxis>
                    <YAxis allowDecimals={false} tick={{ fontSize: 10 }} width={40}>
                      <Label value="Trades" angle={-90} position="insideLeft" fontSize={11} fill="currentColor" />
                    </YAxis>
                    <ChartTooltip
                      content={
                        <ChartTooltipContent
                          formatter={(value) => [`${fmtInt(value as number)}`, ' Trades']}
                        />
                      }
                    />
                    <Bar dataKey="count" name="Trades" radius={[2, 2, 0, 0]}>
                      {(data?.return_distribution ?? []).map((b, i) => (
                        <Cell key={i} fill={b.min >= 0 ? GREEN : RED} />
                      ))}
                      <LabelList dataKey="count" position="top" fontSize={10} fill="#e5e7eb" formatter={(v: number) => fmtInt(v)} />
                    </Bar>
                  </BarChart>
                </ChartContainer>
              )}
            </Panel>

            {/* periods table */}
            <Panel title="PERIOD BREAKDOWN" icon={<BarChart3 className="w-4 h-4" />}>
              {(data?.periods?.length ?? 0) === 0 ? (
                <p className="text-xs text-muted-foreground font-mono">No period data yet.</p>
              ) : (
                <PeriodsTable rows={data?.periods ?? []} />
              )}
            </Panel>

            {/* session + vix breakdown */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <BreakdownCards
                title="SESSION BREAKDOWN"
                icon={<BarChart3 className="w-4 h-4" />}
                testId="session-breakdown"
                data={data?.session_breakdown ?? {}}
              />
              <BreakdownCards
                title="VIX REGIME BREAKDOWN"
                icon={<AlertTriangle className="w-4 h-4" />}
                testId="vix-breakdown"
                data={data?.vix_regime_breakdown ?? {}}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
