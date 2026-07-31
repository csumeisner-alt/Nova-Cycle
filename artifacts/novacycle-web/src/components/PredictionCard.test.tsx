import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PredictionCard } from './PredictionCard';

// ─── helpers ────────────────────────────────────────────────────────────────

function renderCard(name = 'long', label = 'Long Trend') {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <PredictionCard name={name} label={label} />
    </QueryClientProvider>,
  );
}

afterEach(() => vi.restoreAllMocks());

// ─── NO DATA / fallback ─────────────────────────────────────────────────────

describe('PredictionCard – NO DATA fallback', () => {
  it('shows NO DATA badge when fetch fails (network error)', async () => {
    // retry: 1 in the component means one retry (~1 s delay) before error state.
    // Use a generous timeout so the test doesn't race against the retry window.
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('network error'));
    renderCard();
    await waitFor(
      () => expect(screen.getByTestId('badge-signal-long')).toHaveTextContent('NO DATA'),
      { timeout: 4000 },
    );
    expect(screen.getByTestId('text-confidence-long')).toHaveTextContent('—');
  });

  it('shows NO DATA badge when backend returns non-OK status', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'unavailable' }), { status: 503 }),
    );
    renderCard();
    await waitFor(
      () => expect(screen.getByTestId('badge-signal-long')).toHaveTextContent('NO DATA'),
      { timeout: 4000 },
    );
  });

  it('shows NO DATA badge when confidence_percent is missing from payload', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({ trend: 'UP', display_signal: 'BUY BIAS' /* no confidence_percent */ }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    renderCard();
    await waitFor(() =>
      expect(screen.getByTestId('badge-signal-long')).toHaveTextContent('NO DATA'),
    );
  });

  it('shows NO DATA badge when trend is missing from payload', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({ confidence_percent: 80, display_signal: 'BUY BIAS' }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    renderCard();
    await waitFor(() =>
      expect(screen.getByTestId('badge-signal-long')).toHaveTextContent('NO DATA'),
    );
  });

  it('shows NO DATA badge when display_signal is missing from payload', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({ confidence_percent: 80, trend: 'UP' }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    renderCard();
    await waitFor(() =>
      expect(screen.getByTestId('badge-signal-long')).toHaveTextContent('NO DATA'),
    );
  });
});

// ─── zone colours ───────────────────────────────────────────────────────────

describe('PredictionCard – confidence zone colours', () => {
  async function renderWithPct(pct: number) {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          confidence_percent: pct,
          trend: 'NEUTRAL',
          display_signal: 'NEUTRAL / HOLD',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    renderCard();
    await waitFor(() =>
      expect(screen.getByTestId('text-confidence-long')).not.toHaveTextContent('—'),
    );
    return screen.getByTestId('bar-confidence-long');
  }

  it('bar is red at pct=0', async () => {
    const bar = await renderWithPct(0);
    expect(bar.className).toContain('bg-red-400');
  });

  it('bar is red at pct=30 (top of red zone)', async () => {
    const bar = await renderWithPct(30);
    expect(bar.className).toContain('bg-red-400');
  });

  it('bar is yellow at pct=31 (bottom of yellow zone)', async () => {
    const bar = await renderWithPct(31);
    expect(bar.className).toContain('bg-amber-400');
  });

  it('bar is yellow at pct=64 (top of yellow zone)', async () => {
    const bar = await renderWithPct(64);
    expect(bar.className).toContain('bg-amber-400');
  });

  it('bar is green at pct=65 (bottom of green zone)', async () => {
    const bar = await renderWithPct(65);
    expect(bar.className).toContain('bg-emerald-400');
  });

  it('bar is green at pct=100', async () => {
    const bar = await renderWithPct(100);
    expect(bar.className).toContain('bg-emerald-400');
  });
});

// ─── data-quality warning banner ────────────────────────────────────────────

function mockPredictResponse(overrides: Record<string, unknown> = {}) {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(
      JSON.stringify({
        confidence_percent: 72,
        trend: 'UP',
        display_signal: 'BUY BIAS',
        ...overrides,
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ),
  );
}

describe('PredictionCard – data-quality warning banner', () => {
  it('shows the banner when data_quality_degraded is true', async () => {
    mockPredictResponse({
      data_quality_degraded: true,
      data_quality_reason: 'Candle at 2024-01-15 09:30 had zero volume and was filtered.',
    });
    renderCard();
    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-long')).toBeInTheDocument(),
    );
  });

  it('expands to reveal the reason string when the toggle button is clicked', async () => {
    mockPredictResponse({
      data_quality_degraded: true,
      data_quality_reason: 'Candle at 2024-01-15 09:30 had zero volume and was filtered.',
    });
    renderCard();

    // Wait for the banner to appear
    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-long')).toBeInTheDocument(),
    );

    // Reason text is hidden before expanding
    expect(screen.queryByTestId('text-data-quality-reason-long')).not.toBeInTheDocument();

    // Click the expand toggle
    fireEvent.click(screen.getByRole('button', { name: /toggle data quality detail/i }));

    // Reason text should now be visible
    expect(screen.getByTestId('text-data-quality-reason-long')).toHaveTextContent(
      'Candle at 2024-01-15 09:30 had zero volume and was filtered.',
    );
  });

  it('does not show the banner when data_quality_degraded is absent', async () => {
    mockPredictResponse(); // no data_quality_degraded field
    renderCard();
    await waitFor(() =>
      expect(screen.getByTestId('text-confidence-long')).toHaveTextContent('72%'),
    );
    expect(screen.queryByTestId('banner-data-quality-long')).not.toBeInTheDocument();
  });

  it('does not show the banner when data_quality_degraded is false', async () => {
    mockPredictResponse({ data_quality_degraded: false });
    renderCard();
    await waitFor(() =>
      expect(screen.getByTestId('text-confidence-long')).toHaveTextContent('72%'),
    );
    expect(screen.queryByTestId('banner-data-quality-long')).not.toBeInTheDocument();
  });
});

// ─── cross_bar_spike quarantine banner ──────────────────────────────────────

describe('PredictionCard – cross_bar_spike quarantine banner', () => {
  const SPIKE_REASON = 'quarantined 1 malformed daily candle(s); latest bad candle ts=2024-07-30T00:00:00 reason=cross_bar_spike (close=421.0500); using last valid candle instead';

  it('shows the spike-specific summary message for a cross_bar_spike reason (long-trend card)', async () => {
    mockPredictResponse({
      data_quality_degraded: true,
      data_quality_reason: SPIKE_REASON,
    });
    renderCard('long', 'Long Trend');

    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-long')).toBeInTheDocument(),
    );

    const banner = screen.getByTestId('banner-data-quality-long');
    expect(banner).toHaveTextContent('Glitch bar quarantined');
    expect(banner).toHaveTextContent('price spike was detected and excluded');
  });

  it('shows the spike-specific summary message for a cross_bar_spike reason (short-trend card)', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          confidence_percent: 58,
          trend: 'DOWN',
          display_signal: 'SELL BIAS',
          data_quality_degraded: true,
          data_quality_reason: SPIKE_REASON,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    renderCard('short', 'Short Trend');

    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-short')).toBeInTheDocument(),
    );

    const banner = screen.getByTestId('banner-data-quality-short');
    expect(banner).toHaveTextContent('Glitch bar quarantined');
    expect(banner).toHaveTextContent('price spike was detected and excluded');
  });

  it('does NOT show spike-specific message for a generic (non-spike) quarantine reason', async () => {
    mockPredictResponse({
      data_quality_degraded: true,
      data_quality_reason: 'quarantined 1 malformed daily candle(s); reason=high_below_open',
    });
    renderCard('long', 'Long Trend');

    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-long')).toBeInTheDocument(),
    );

    const banner = screen.getByTestId('banner-data-quality-long');
    expect(banner).not.toHaveTextContent('Glitch bar quarantined');
    expect(banner).toHaveTextContent('one or more candles were filtered');
  });

  it('expands to reveal the raw cross_bar_spike reason string', async () => {
    mockPredictResponse({
      data_quality_degraded: true,
      data_quality_reason: SPIKE_REASON,
    });
    renderCard('long', 'Long Trend');

    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-long')).toBeInTheDocument(),
    );

    // Reason text hidden before expanding
    expect(screen.queryByTestId('text-data-quality-reason-long')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /toggle data quality detail/i }));

    expect(screen.getByTestId('text-data-quality-reason-long')).toHaveTextContent(SPIKE_REASON);
  });

  it('spike-quarantine banner appears on short-trend card after auto-refresh that coincides with a spike', async () => {
    // First fetch clean; second (simulated refetch) returns spike quarantine
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ confidence_percent: 58, trend: 'NEUTRAL', display_signal: 'NEUTRAL / HOLD' }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            confidence_percent: 58,
            trend: 'NEUTRAL',
            display_signal: 'NEUTRAL / HOLD',
            data_quality_degraded: true,
            data_quality_reason: SPIKE_REASON,
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      );

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <PredictionCard name="short" label="Short Trend" />
      </QueryClientProvider>,
    );

    // First fetch – no banner
    await waitFor(() =>
      expect(screen.getByTestId('text-confidence-short')).toHaveTextContent('58%'),
    );
    expect(screen.queryByTestId('banner-data-quality-short')).not.toBeInTheDocument();

    // Simulate the auto-refresh interval firing
    await act(async () => {
      await qc.refetchQueries({ queryKey: ['predict', 'short'] });
    });

    // Banner must appear after the spike-quarantine refetch
    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-short')).toBeInTheDocument(),
    );
    expect(screen.getByTestId('banner-data-quality-short')).toHaveTextContent('Glitch bar quarantined');
  });
});

// ─── auto-refresh transition tests ──────────────────────────────────────────

function makeResponse(overrides: Record<string, unknown> = {}) {
  return new Response(
    JSON.stringify({
      confidence_percent: 72,
      trend: 'UP',
      display_signal: 'BUY BIAS',
      ...overrides,
    }),
    { status: 200, headers: { 'Content-Type': 'application/json' } },
  );
}

describe('PredictionCard – data-quality banner auto-refresh transitions', () => {
  it('banner disappears after a clean refetch following a degraded response', async () => {
    // First fetch returns degraded; second (refetch) returns clean
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(makeResponse({ data_quality_degraded: true, data_quality_reason: 'Zero volume candle filtered.' }))
      .mockResolvedValueOnce(makeResponse({ data_quality_degraded: false }));

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <PredictionCard name="long" label="Long Trend" />
      </QueryClientProvider>,
    );

    // Banner is present after the first (degraded) fetch
    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-long')).toBeInTheDocument(),
    );

    // Trigger a refetch manually (simulates the 60 s interval firing)
    await act(async () => {
      await qc.refetchQueries({ queryKey: ['predict', 'long'] });
    });

    // Banner must be gone after the clean refetch
    await waitFor(() =>
      expect(screen.queryByTestId('banner-data-quality-long')).not.toBeInTheDocument(),
    );
  });

  it('banner appears after a degraded refetch following a clean response', async () => {
    // First fetch returns clean; second (refetch) returns degraded
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(makeResponse({ data_quality_degraded: false }))
      .mockResolvedValueOnce(makeResponse({ data_quality_degraded: true, data_quality_reason: 'Zero volume candle filtered.' }));

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <PredictionCard name="long" label="Long Trend" />
      </QueryClientProvider>,
    );

    // No banner after the first (clean) fetch
    await waitFor(() =>
      expect(screen.getByTestId('text-confidence-long')).toHaveTextContent('72%'),
    );
    expect(screen.queryByTestId('banner-data-quality-long')).not.toBeInTheDocument();

    // Trigger a refetch manually (simulates the 60 s interval firing)
    await act(async () => {
      await qc.refetchQueries({ queryKey: ['predict', 'long'] });
    });

    // Banner must appear after the degraded refetch
    await waitFor(() =>
      expect(screen.getByTestId('banner-data-quality-long')).toBeInTheDocument(),
    );
  });
});
