import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrainingStuckBanner, stuckEpisodeSignature } from '../TrainingStuckBanner';

function mockHealthz(models: Record<string, unknown>) {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(JSON.stringify({ status: 'degraded', models }), { status: 200 }),
  );
}

function renderBanner() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <TrainingStuckBanner />
    </QueryClientProvider>,
  );
}

beforeEach(() => sessionStorage.clear());
afterEach(() => vi.restoreAllMocks());

const STUCK = {
  training_stuck: true,
  consecutive_training_failures: 3,
  last_training_attempted_at: '2026-08-01T00:00:00Z',
};
const HEALTHY = { training_stuck: false, consecutive_training_failures: 0 };

describe('TrainingStuckBanner', () => {
  it('renders nothing when no model is stuck', async () => {
    mockHealthz({ long_trend: HEALTHY, short_trend: HEALTHY });
    renderBanner();
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    expect(screen.queryByTestId('banner-training-stuck')).toBeNull();
  });

  it('shows the banner naming the stuck model', async () => {
    mockHealthz({ long_trend: STUCK, short_trend: HEALTHY });
    renderBanner();
    const banner = await screen.findByTestId('banner-training-stuck');
    expect(banner).toHaveTextContent('LONG TREND');
    expect(banner).toHaveTextContent('training-stuck');
  });

  it('hides after dismissal and persists dismissal in sessionStorage', async () => {
    mockHealthz({ long_trend: STUCK });
    renderBanner();
    await screen.findByTestId('banner-training-stuck');
    fireEvent.click(screen.getByTestId('button-dismiss-training-stuck'));
    expect(screen.queryByTestId('banner-training-stuck')).toBeNull();
    expect(sessionStorage.getItem('training-stuck-banner-dismissed')).toBe(
      stuckEpisodeSignature({ long_trend: STUCK }),
    );
  });

  it('stays hidden on remount for the same episode (per-session dismissal)', async () => {
    sessionStorage.setItem(
      'training-stuck-banner-dismissed',
      stuckEpisodeSignature({ long_trend: STUCK }),
    );
    mockHealthz({ long_trend: STUCK });
    renderBanner();
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled());
    expect(screen.queryByTestId('banner-training-stuck')).toBeNull();
  });

  it('re-appears when a new stuck episode starts after dismissal', async () => {
    // Dismissed signature is for the old episode (3 failures).
    sessionStorage.setItem(
      'training-stuck-banner-dismissed',
      stuckEpisodeSignature({ long_trend: STUCK }),
    );
    // New episode: another failed retrain attempt bumps the signature.
    mockHealthz({
      long_trend: {
        ...STUCK,
        consecutive_training_failures: 4,
        last_training_attempted_at: '2026-08-02T00:00:00Z',
      },
    });
    renderBanner();
    expect(await screen.findByTestId('banner-training-stuck')).toBeInTheDocument();
  });

  it('lists multiple stuck models', async () => {
    mockHealthz({ long_trend: STUCK, short_trend: STUCK });
    renderBanner();
    const banner = await screen.findByTestId('banner-training-stuck');
    expect(banner).toHaveTextContent('LONG TREND');
    expect(banner).toHaveTextContent('SHORT TREND');
    expect(banner).toHaveTextContent('are training-stuck');
  });
});
