import { describe, it, expect } from 'vitest';
import { confidenceZone } from './confidenceZone';

describe('confidenceZone – zone boundaries', () => {
  // ── Red zone: 0–30 ──────────────────────────────────────────────────────
  it('returns red at 0 (minimum)', () => {
    const z = confidenceZone(0);
    expect(z.text).toBe('text-red-400');
    expect(z.bar).toBe('bg-red-400');
  });

  it('returns red at 30 (top of red zone)', () => {
    const z = confidenceZone(30);
    expect(z.text).toBe('text-red-400');
    expect(z.bar).toBe('bg-red-400');
  });

  // ── Yellow zone: 31–64 ──────────────────────────────────────────────────
  it('returns yellow at 31 (bottom of yellow zone)', () => {
    const z = confidenceZone(31);
    expect(z.text).toBe('text-amber-400');
    expect(z.bar).toBe('bg-amber-400');
  });

  it('returns yellow at 64 (top of yellow zone)', () => {
    const z = confidenceZone(64);
    expect(z.text).toBe('text-amber-400');
    expect(z.bar).toBe('bg-amber-400');
  });

  // ── Green zone: 65–100 ──────────────────────────────────────────────────
  it('returns green at 65 (bottom of green zone)', () => {
    const z = confidenceZone(65);
    expect(z.text).toBe('text-emerald-400');
    expect(z.bar).toBe('bg-emerald-400');
  });

  it('returns green at 100 (maximum)', () => {
    const z = confidenceZone(100);
    expect(z.text).toBe('text-emerald-400');
    expect(z.bar).toBe('bg-emerald-400');
  });

  // ── Shape sanity ────────────────────────────────────────────────────────
  it('returns all four keys for every zone', () => {
    for (const pct of [0, 30, 31, 64, 65, 100]) {
      const z = confidenceZone(pct);
      expect(z).toHaveProperty('text');
      expect(z).toHaveProperty('bar');
      expect(z).toHaveProperty('border');
      expect(z).toHaveProperty('bg');
    }
  });
});
