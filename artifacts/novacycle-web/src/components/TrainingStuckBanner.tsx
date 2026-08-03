import { useQuery } from '@tanstack/react-query';
import { AlertTriangle, X } from 'lucide-react';
import { useState } from 'react';

/**
 * Site-wide sticky banner shown when any model reports training_stuck in
 * /api/healthz. Reuses the shared ['healthz'] query so no extra polling
 * loop is added (the StatusDashboard already refetches every 5 s).
 *
 * Dismissal is per session (sessionStorage) and keyed by an episode
 * signature: the stuck model names plus each model's last training attempt
 * timestamp / failure count. A NEW stuck episode (different models, or a
 * further failed retrain attempt) produces a new signature, so the banner
 * re-appears even after a previous dismissal.
 */

type HealthzModel = {
  training_stuck?: boolean;
  consecutive_training_failures?: number;
  last_training_attempted_at?: string | null;
};

const DISMISS_KEY = 'training-stuck-banner-dismissed';

export function stuckEpisodeSignature(models: Record<string, HealthzModel>): string {
  return Object.entries(models)
    .filter(([, m]) => m?.training_stuck === true)
    .map(
      ([name, m]) =>
        `${name}:${m.consecutive_training_failures ?? 0}:${m.last_training_attempted_at ?? ''}`,
    )
    .sort()
    .join('|');
}

function readDismissed(): string | null {
  try {
    return sessionStorage.getItem(DISMISS_KEY);
  } catch {
    return null;
  }
}

export function TrainingStuckBanner() {
  // Track dismissal in state so the banner hides immediately on click;
  // sessionStorage keeps it hidden across route changes within the session.
  const [dismissedSig, setDismissedSig] = useState<string | null>(readDismissed);

  const { data: health } = useQuery<any>({
    queryKey: ['healthz'],
    queryFn: async () => {
      const res = await fetch('/api/healthz');
      if (!res.ok) throw new Error('Backend unreachable');
      return res.json();
    },
    refetchInterval: 5000,
    retry: 2,
  });

  const models: Record<string, HealthzModel> = health?.models ?? {};
  const stuckModels = Object.entries(models)
    .filter(([, m]) => m?.training_stuck === true)
    .map(([name]) => name);

  if (stuckModels.length === 0) return null;

  const signature = stuckEpisodeSignature(models);
  if (dismissedSig === signature) return null;

  const dismiss = () => {
    setDismissedSig(signature);
    try {
      sessionStorage.setItem(DISMISS_KEY, signature);
    } catch {
      /* sessionStorage unavailable — dismissal lasts for this render only */
    }
  };

  const modelLabels = stuckModels.map((n) => n.replace('_', ' ').toUpperCase()).join(', ');

  return (
    <div
      className="sticky top-0 z-50 w-full border-b border-red-400/30 bg-red-950/95 backdrop-blur-md"
      role="alert"
      data-testid="banner-training-stuck"
    >
      <div className="mx-auto max-w-7xl px-6 py-2.5 flex items-center gap-3">
        <AlertTriangle className="w-4 h-4 shrink-0 text-red-400" />
        <p className="flex-1 text-xs sm:text-sm font-mono text-red-200 leading-snug">
          <span className="font-medium text-red-400">MODEL DEGRADED — </span>
          {modelLabels} {stuckModels.length > 1 ? 'are' : 'is'} training-stuck (repeated retrain
          failures) and running on a stale model. Signals from{' '}
          {stuckModels.length > 1 ? 'these models' : 'this model'} are not reliable.
        </p>
        <button
          type="button"
          onClick={dismiss}
          aria-label="Dismiss training-stuck warning"
          data-testid="button-dismiss-training-stuck"
          className="p-1 rounded text-red-300 hover:text-red-100 hover:bg-red-400/20 transition-colors shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
