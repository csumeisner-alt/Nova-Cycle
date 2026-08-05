import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DrawdownGatePanel } from '@/App';

// DrawdownGatePanel takes { health: any } and reads health?.drawdown_gate.
// It has three rendering modes:
//   undefined  → key absent (older backend) → returns null (renders nothing)
//   null       → key present but no file found → "not run yet" state
//   object     → valid report → full gate panel

describe('DrawdownGatePanel', () => {
  it('renders nothing when drawdown_gate key is absent from health (undefined)', () => {
    // health object has no drawdown_gate key at all — older backend compatibility.
    const { container } = render(<DrawdownGatePanel health={{}} />);
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId('panel-drawdown-gate')).toBeNull();
  });

  it('renders nothing when the entire health object is null', () => {
    // Defensive: if health itself is null, drawdown_gate resolves to undefined.
    const { container } = render(<DrawdownGatePanel health={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the "not run yet" state when drawdown_gate is explicitly null', () => {
    // File is absent on disk → backend returns drawdown_gate: null.
    render(<DrawdownGatePanel health={{ drawdown_gate: null }} />);

    const panel = screen.getByTestId('panel-drawdown-gate');
    expect(panel).toBeInTheDocument();

    // Must show the instructional "not run yet" message.
    const notRunMsg = screen.getByTestId('drawdown-gate-not-run');
    expect(notRunMsg).toBeInTheDocument();
    expect(notRunMsg).toHaveTextContent(/no dry-run report found/i);
  });

  it('does not show the full gate metrics when drawdown_gate is null', () => {
    render(<DrawdownGatePanel health={{ drawdown_gate: null }} />);

    // These testids only appear when a real report is present.
    expect(screen.queryByTestId('drawdown-gate-badge')).toBeNull();
    expect(screen.queryByTestId('drawdown-gate-pr-auc-lift')).toBeNull();
  });

  it('renders the full panel when a valid report is present', () => {
    const report = {
      run_timestamp_utc: '2026-07-15T10:00:00',
      data_source: 'yfinance',
      total_configs_evaluated: 8,
      configs_passing_gate: 0,
      promotion_gate_description: 'PR-AUC_lift>=2 AND precision_lift>=2',
      best_result: {
        label: 'h5_dd0.05_xgb',
        passes_promotion_gate: false,
        pr_auc_lift_vs_prevalence: 1.2,
        precision_lift_vs_base_rate: 1.1,
        avoided_drawdown_recall: 0.4,
        positive_rate: 0.08,
      },
      passing_results: [],
    };

    render(<DrawdownGatePanel health={{ drawdown_gate: report }} />);

    expect(screen.getByTestId('panel-drawdown-gate')).toBeInTheDocument();
    expect(screen.getByTestId('drawdown-gate-badge')).toBeInTheDocument();
    expect(screen.getByTestId('drawdown-gate-pr-auc-lift')).toBeInTheDocument();
    // "not run yet" message must NOT appear when report is present.
    expect(screen.queryByTestId('drawdown-gate-not-run')).toBeNull();
  });
});
