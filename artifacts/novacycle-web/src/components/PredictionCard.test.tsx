import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
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
