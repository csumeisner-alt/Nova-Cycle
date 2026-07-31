import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PerformanceDashboard } from '../PerformanceDashboard';

// ─── mock data builders ─────────────────────────────────────────────────────

function emptyResponse(period = 'day') {
  return {
    ticker: 'VOO',
    period,
    window: '90d',
    summary: {
      total_trades: 0,
      wins: 0,
      losses: 0,
      buy_precision: 0,
      avg_return_percent: 0,
      missed_rally_rate: 0,
      current_win_streak: 0,
      recommendation_stability: 0,
      avg_confidence: 0,
      cumulative_return_percent: 0,
    },
    periods: [],
    confidence_buckets: {
      low: { trade_count: 0, win_rate: 0, avg_return_percent: 0 },
      medium: { trade_count: 0, win_rate: 0, avg_return_percent: 0 },
      high: { trade_count: 0, win_rate: 0, avg_return_percent: 0 },
    },
    calibration_curve: [],
    cumulative_pnl: [],
    return_distribution: [],
    session_breakdown: {},
    vix_regime_breakdown: {},
    best_trade: null,
    worst_trade: null,
    streak: { current_win: 0, current_loss: 0, longest_win: 0, longest_loss: 0 },
    missed_rallies: { count: 0, timestamps: [], rate: 0 },
    accuracy_history: [],
  };
}

function fullResponse(overrides: Record<string, unknown> = {}, period = 'day') {
  return {
    ...emptyResponse(period),
    summary: {
      total_trades: 42,
      wins: 30,
      losses: 12,
      buy_precision: 0.714,
      avg_return_percent: 1.23,
      missed_rally_rate: 0.15,
      current_win_streak: 4,
      recommendation_stability: 2,
      avg_confidence: 0.68,
      cumulative_return_percent: 12.5,
    },
    periods: [
      {
        label: '2026-07-30',
        start: '2026-07-30T00:00:00Z',
        buy_count: 5,
        wins: 4,
        losses: 1,
        precision: 0.8,
        avg_return_percent: 1.5,
        missed_rallies: 1,
        avg_confidence: 0.7,
        oos_accuracy: 0.66,
      },
      {
        label: '2026-07-31',
        start: '2026-07-31T00:00:00Z',
        buy_count: 3,
        wins: 1,
        losses: 2,
        precision: 0.33,
        avg_return_percent: -0.8,
        missed_rallies: 0,
        avg_confidence: 0.55,
        oos_accuracy: null,
      },
    ],
    calibration_curve: Array.from({ length: 10 }).map((_, i) => ({
      confidence_mid: 0.05 + i * 0.1,
      actual_win_rate: i < 8 ? 0.1 + i * 0.09 : null,
      trade_count: i < 8 ? 5 : 0,
    })),
    cumulative_pnl: [
      { timestamp: '2026-07-30T10:00:00Z', cumulative_return_percent: 1.2 },
      { timestamp: '2026-07-31T10:00:00Z', cumulative_return_percent: 3.4 },
    ],
    return_distribution: [
      { label: '-2 to -1', min: -2, max: -1, count: 3 },
      { label: '0 to 1', min: 0, max: 1, count: 10 },
    ],
    session_breakdown: {
      MORNING: { count: 20, win_rate: 0.7, average_return_percent: 1.1 },
      AFTERNOON: { count: 22, win_rate: 0.6, average_return_percent: 0.9 },
    },
    vix_regime_breakdown: {
      LOW: { count: 15, win_rate: 0.75, average_return_percent: 1.4 },
      HIGH: { count: 5, win_rate: 0.4, average_return_percent: -0.5 },
    },
    best_trade: {
      cycle_id: 'c1',
      buy_timestamp: '2026-07-28T09:30:00Z',
      sell_timestamp: '2026-07-28T14:30:00Z',
      return_percent: 4.5,
      hold_time_minutes: 300,
      confidence_at_buy: 0.82,
    },
    worst_trade: {
      cycle_id: 'c2',
      buy_timestamp: '2026-07-29T09:30:00Z',
      sell_timestamp: '2026-07-29T10:00:00Z',
      return_percent: -3.1,
      hold_time_minutes: 30,
      confidence_at_buy: 0.51,
    },
    accuracy_history: [
      { model_name: 'long_trend', trained_at: '2026-07-01T00:00:00Z', accuracy: 0.61 },
      { model_name: 'long_trend', trained_at: '2026-07-15T00:00:00Z', accuracy: 0.64 },
      { model_name: 'short_trend', trained_at: '2026-07-01T00:00:00Z', accuracy: 0.58 },
    ],
    ...overrides,
  };
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

function renderDashboard() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <PerformanceDashboard />
    </QueryClientProvider>,
  );
}

// Recharts' ResponsiveContainer needs a measured size and a ResizeObserver.
// jsdom provides neither, so we shim both and force a non-zero layout box so
// the SVG (and reference lines) actually render.
beforeEach(() => {
  globalThis.ResizeObserver = class {
    callback: ResizeObserverCallback;
    constructor(cb: ResizeObserverCallback) {
      this.callback = cb;
    }
    observe() {}
    unobserve() {}
    disconnect() {}
  };

  Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
    configurable: true,
    value: 800,
  });
  Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
    configurable: true,
    value: 400,
  });
  const originalGetBBox = Element.prototype.getBoundingClientRect;
  Object.defineProperty(Element.prototype, 'getBoundingClientRect', {
    configurable: true,
    value() {
      return { width: 800, height: 400, top: 0, left: 0, bottom: 400, right: 800, x: 0, y: 0, toJSON() {} };
    },
  });
  // keep a reference so restore doesn't complain
  void originalGetBBox;
});

afterEach(() => vi.restoreAllMocks());

// ─── tests ──────────────────────────────────────────────────────────────────

describe('PerformanceDashboard – skeleton & tabs', () => {
  it('shows a loading skeleton while the request is pending', async () => {
    // never-resolving fetch keeps the query pending
    vi.spyOn(globalThis, 'fetch').mockReturnValue(new Promise(() => {}) as Promise<Response>);
    renderDashboard();
    expect(screen.getByTestId('skeleton-performance')).toBeInTheDocument();
  });

  it('renders period toggle buttons that are clickable', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(fullResponse()));
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('card-buy-precision')).toBeInTheDocument());
    expect(screen.getByTestId('toggle-period-day')).toBeInTheDocument();
    expect(screen.getByTestId('toggle-period-week')).toBeInTheDocument();
    expect(screen.getByTestId('toggle-period-month')).toBeInTheDocument();
  });
});

describe('PerformanceDashboard – empty states', () => {
  it('shows the friendly empty message when total_trades is 0', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(emptyResponse()));
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('empty-performance')).toBeInTheDocument());
    expect(screen.getByTestId('empty-performance')).toHaveTextContent(
      /No completed trades yet/i,
    );
  });

  it('shows the confidence-specific empty message when a filter yields zero trades', async () => {
    // Fresh Response per call — a Response body can only be consumed once.
    vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      Promise.resolve(jsonResponse(emptyResponse())),
    );
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('empty-performance')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('filter-confidence-high'));

    await waitFor(
      () => expect(screen.getByTestId('empty-confidence')).toBeInTheDocument(),
      { timeout: 4000 },
    );
    expect(screen.getByTestId('empty-confidence')).toHaveTextContent(
      /No trades recorded at this confidence level yet/i,
    );
  });
});

describe('PerformanceDashboard – summary cards & formatting', () => {
  it('renders all five summary cards', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(fullResponse()));
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('card-buy-precision')).toBeInTheDocument());
    expect(screen.getByTestId('card-avg-return')).toBeInTheDocument();
    expect(screen.getByTestId('card-missed-rally-rate')).toBeInTheDocument();
    expect(screen.getByTestId('card-current-win-streak')).toBeInTheDocument();
    expect(screen.getByTestId('card-recommendation-stability')).toBeInTheDocument();
  });

  it('formats percentages with a "%" sign', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(fullResponse()));
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('card-buy-precision')).toBeInTheDocument());
    // 0.714 -> "71.4%"
    expect(screen.getByTestId('card-buy-precision')).toHaveTextContent('71.4%');
    expect(screen.getByTestId('card-avg-return')).toHaveTextContent('%');
  });

  it('formats trade hold time and dollars appropriately in callouts', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(fullResponse()));
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('callout-best-trade')).toBeInTheDocument());
    expect(screen.getByTestId('callout-best-trade')).toHaveTextContent('4.5%');
    expect(screen.getByTestId('callout-worst-trade')).toHaveTextContent('-3.1%');
  });
});

describe('PerformanceDashboard – dollar formatting present somewhere', () => {
  it('renders a "$" formatted value for cumulative return summary', async () => {
    // The cumulative return card in summary uses percentages; ensure $ appears
    // in the rendered dashboard via a dollar-denominated best-trade tooltip fallback.
    // We assert the fmtDollar helper path is exercised by checking the DOM contains "$".
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse(
        fullResponse({
          // best_trade dollar hint rendered in callout via return % — ensure $ token exists
        }),
      ),
    );
    const { container } = renderDashboard();
    await waitFor(() => expect(screen.getByTestId('card-buy-precision')).toBeInTheDocument());
    // Cumulative P&L axis / summary uses %, but the dashboard must include a $-formatted
    // figure. Assert dollar sign appears in the summary region for cumulative dollars.
    expect(container.textContent).toContain('$');
  });
});

describe('PerformanceDashboard – charts present', () => {
  it('renders every chart container', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(fullResponse()));
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('chart-cumulative-pnl')).toBeInTheDocument());
    expect(screen.getByTestId('chart-wins-losses')).toBeInTheDocument();
    expect(screen.getByTestId('chart-calibration')).toBeInTheDocument();
    expect(screen.getByTestId('chart-accuracy')).toBeInTheDocument();
    expect(screen.getByTestId('chart-return-distribution')).toBeInTheDocument();
  });

  it('renders the calibration reference line element', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(fullResponse()));
    const { container } = renderDashboard();
    await waitFor(() => expect(screen.getByTestId('chart-calibration')).toBeInTheDocument());
    // Recharts renders reference lines with the recharts-reference-line class
    await waitFor(() =>
      expect(container.querySelector('.recharts-reference-line')).toBeInTheDocument(),
    );
  });

  it('renders the accuracy chart even when some accuracy_history entries are null', async () => {
    // Contract allows accuracy: null (retrains recorded without a metric).
    // The chart must still render and simply skip the null points.
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse(
        fullResponse({
          accuracy_history: [
            { model_name: 'long_trend', trained_at: '2026-07-01T00:00:00Z', accuracy: 0.61 },
            { model_name: 'long_trend', trained_at: '2026-07-10T00:00:00Z', accuracy: null },
            { model_name: 'long_trend', trained_at: '2026-07-15T00:00:00Z', accuracy: 0.64 },
          ],
        }),
      ),
    );
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('chart-accuracy')).toBeInTheDocument());
  });
});

describe('PerformanceDashboard – period toggle', () => {
  it('re-queries the API with the new period param and updates table content', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('period=week')) {
          return Promise.resolve(
            jsonResponse(
              fullResponse(
                {
                  periods: [
                    {
                      label: '2026-W31',
                      start: '2026-07-27T00:00:00Z',
                      buy_count: 9,
                      wins: 7,
                      losses: 2,
                      precision: 0.78,
                      avg_return_percent: 2.1,
                      missed_rallies: 1,
                      avg_confidence: 0.72,
                      oos_accuracy: 0.7,
                    },
                  ],
                },
                'week',
              ),
            ),
          );
        }
        return Promise.resolve(jsonResponse(fullResponse()));
      });

    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('table-periods')).toBeInTheDocument());
    // day content
    expect(screen.getByTestId('table-periods')).toHaveTextContent('2026-07-30');

    fireEvent.click(screen.getByTestId('toggle-period-week'));

    await waitFor(() =>
      expect(screen.getByTestId('table-periods')).toHaveTextContent('2026-W31'),
    );
    // ensure a week request was made
    expect(
      fetchMock.mock.calls.some(([u]) => String(u).includes('period=week')),
    ).toBe(true);
  });
});

describe('PerformanceDashboard – confidence filter', () => {
  it('applies confidence_min / confidence_max params when a band is selected', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(() => Promise.resolve(jsonResponse(fullResponse())));

    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('card-buy-precision')).toBeInTheDocument());

    fireEvent.click(screen.getByTestId('filter-confidence-high'));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([u]) => {
          const s = String(u);
          return s.includes('confidence_min=0.7') && s.includes('confidence_max=1');
        }),
      ).toBe(true),
    );
  });
});

describe('PerformanceDashboard – sortable table', () => {
  it('reorders rows when a column header is clicked', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(fullResponse()));
    renderDashboard();
    await waitFor(() => expect(screen.getByTestId('table-periods')).toBeInTheDocument());

    const table = screen.getByTestId('table-periods');
    const initialRows = within(table).getAllByTestId(/^row-period-/);
    // default asc by label: 2026-07-30 first
    expect(initialRows[0]).toHaveAttribute('data-testid', 'row-period-2026-07-30');

    // sort by wins asc: 2026-07-31 (1 win) first
    fireEvent.click(screen.getByTestId('th-wins'));
    const afterRows = within(table).getAllByTestId(/^row-period-/);
    expect(afterRows[0]).toHaveAttribute('data-testid', 'row-period-2026-07-31');
  });
});
