import { useQuery } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, ChevronDown, ChevronUp, Star, Zap } from 'lucide-react';
import { confidenceZone } from '@/lib/confidenceZone';
import { useState } from 'react';

export type PredictionDisplay = {
  confidence_percent?: number;
  trend?: 'UP' | 'DOWN' | 'NEUTRAL';
  display_signal?: 'BUY BIAS' | 'SELL BIAS' | 'NEUTRAL / HOLD';
  note?: string;
  data_quality_degraded?: boolean;
  data_quality_reason?: string;
  conviction_tier?: 'opportunity' | 'high_conviction' | null;
  conviction_reasons?: string[];
  /**
   * True when the decision filter soft-blocked the raw gauge direction.
   * The signal is informational only — not executable, never stored in
   * history, never push-notified.
   */
  is_candidate?: boolean;
  /**
   * The raw gauge direction when is_candidate is true ("buy" or "sell").
   * Null for normal (non-candidate) responses.
   */
  candidate_signal?: 'buy' | 'sell' | null;
  /** Conviction tier for the candidate direction (always "opportunity" or null). */
  candidate_conviction_tier?: 'opportunity' | null;
  /**
   * Explicit model availability semantics: a neutral result from a stale or
   * training-stuck model must never look like a healthy recommendation.
   */
  model_state?: 'healthy' | 'model_unavailable' | 'training_stuck' | 'stale_rolled_back' | 'baseline_mode';
  prediction_reliable?: boolean;
  /** "trained" when a gate-passing model is active; "baseline" when the signal
   *  falls back to the calibrated majority-class base rate (~73% bull bias). */
  long_signal_mode?: 'trained' | 'baseline';
};

const MODEL_STATE_SUMMARIES: Record<string, string> = {
  model_unavailable: 'MODEL UNAVAILABLE',
  training_stuck: 'MODEL DEGRADED · STALE MODEL',
  stale_rolled_back: 'MODEL STALE · ROLLED BACK',
  baseline_mode: 'BASELINE MODE · NO TRAINED EDGE',
};

const MODEL_STATE_LABELS: Record<string, string> = {
  model_unavailable: 'MODEL UNAVAILABLE — neutral fallback, not a real prediction.',
  training_stuck:
    'MODEL DEGRADED — repeated retraining failures; running on a stale model. Do not treat this as a reliable signal.',
  stale_rolled_back:
    'MODEL STALE — last retrain failed and was rolled back. Signal reliability is reduced.',
  baseline_mode:
    'BASELINE MODE — no trained model passes the OOS quality gate. Showing calibrated base rate (~73% bull bias). This is not a trained directional prediction.',
};

function ConvictionBadge({ tier, name }: { tier: NonNullable<PredictionDisplay['conviction_tier']>; name: string }) {
  const isHigh = tier === 'high_conviction';
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono border ${
        isHigh
          ? 'text-amber-300 bg-amber-400/10 border-amber-400/30'
          : 'text-muted-foreground bg-white/5 border-white/10'
      }`}
      data-testid={`badge-conviction-${name}`}
    >
      {isHigh && <Star className="w-3 h-3 fill-current" />}
      <span>{isHigh ? 'HIGH-CONVICTION' : 'OPPORTUNITY'}</span>
    </span>
  );
}

/**
 * Badge shown when a signal is a candidate — raw gauge crossed its threshold
 * in the given direction but current conditions make it non-executable.
 * Candidates are never stored in history and never push-notify.
 */
function CandidateBadge({ direction, name }: { direction: string; name: string }) {
  const dirLabel = direction.toUpperCase();
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono border text-amber-400/80 bg-amber-400/8 border-amber-400/25"
      data-testid={`badge-candidate-${name}`}
      title="This signal passed the raw gauge but was soft-blocked by the decision filter. It is informational only — not an executable signal."
    >
      <Zap className="w-3 h-3" />
      <span>{dirLabel} CANDIDATE</span>
    </span>
  );
}

function TrendArrow({ trend }: { trend: PredictionDisplay['trend'] }) {
  if (trend === 'UP') return <TrendingUp className="w-4 h-4 text-emerald-400" data-testid="icon-trend-up" />;
  if (trend === 'DOWN') return <TrendingDown className="w-4 h-4 text-red-400" data-testid="icon-trend-down" />;
  return <Minus className="w-4 h-4 text-muted-foreground" data-testid="icon-trend-neutral" />;
}

export function PredictionCard({ name, label }: { name: string; label: string }) {
  const [showReason, setShowReason] = useState(false);
  const [showModelDetails, setShowModelDetails] = useState(false);

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
  const isSpikeQuarantine = degradedReason.includes('cross_bar_spike');

  const bannerSummary = isSpikeQuarantine
    ? '⚠ Glitch bar quarantined: a price spike was detected and excluded. Signal may be less reliable.'
    : '⚠ Data quality issue: one or more candles were filtered. Signal may be less reliable.';

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
      {data?.is_candidate && data?.candidate_signal ? (
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <CandidateBadge direction={data.candidate_signal} name={name} />
          </div>
          <p
            className="text-[11px] font-mono text-amber-400/60 leading-relaxed"
            data-testid={`text-candidate-note-${name}`}
          >
            Direction noted — not executable. Strong continuation gap prevents action.
          </p>
        </div>
      ) : data?.conviction_tier ? (
        <div className="flex items-center gap-2" title={(data.conviction_reasons ?? []).join(' ')}>
          <ConvictionBadge tier={data.conviction_tier} name={name} />
        </div>
      ) : null}
      <div className="h-2 w-full rounded-full bg-white/5 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${zone.bar}`}
          style={{ width: `${pct}%` }}
          data-testid={`bar-confidence-${name}`}
        />
      </div>
      {data?.prediction_reliable === false &&
        data?.model_state &&
        MODEL_STATE_LABELS[data.model_state] &&
        MODEL_STATE_SUMMARIES[data.model_state] && (
        <div
          className="rounded-md border border-red-400/30 bg-red-400/5"
          data-testid={`banner-model-state-${name}`}
        >
          <button
            type="button"
            onClick={() => setShowModelDetails((v) => !v)}
            className="flex w-full items-center justify-between gap-2 px-2.5 py-1.5 text-left text-red-400"
            aria-expanded={showModelDetails}
            aria-label={`Toggle model reliability detail for ${label}`}
          >
            <span className="flex min-w-0 items-center gap-1.5 text-[10px] font-mono font-medium tracking-wide">
              <AlertTriangle className="h-3 w-3 shrink-0" />
              <span className="truncate">{MODEL_STATE_SUMMARIES[data.model_state]}</span>
            </span>
            {showModelDetails ? (
              <ChevronUp className="h-3 w-3 shrink-0" />
            ) : (
              <ChevronDown className="h-3 w-3 shrink-0" />
            )}
          </button>
          {showModelDetails && (
            <p
              className="border-t border-red-400/20 px-2.5 py-2 text-[11px] font-mono leading-relaxed text-red-200/75"
              data-testid={`text-model-state-detail-${name}`}
            >
              {MODEL_STATE_LABELS[data.model_state]}
            </p>
          )}
        </div>
      )}
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
              {bannerSummary}
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
