/**
 * Maps a 0–100 confidence percentage to Tailwind colour tokens.
 *
 * Zones:
 *  - green  (emerald)  : 65 – 100
 *  - yellow (amber)    : 31 –  64
 *  - red               :  0 –  30
 */
export type ConfidenceZone = {
  text: string;
  bar: string;
  border: string;
  bg: string;
};

export function confidenceZone(pct: number): ConfidenceZone {
  if (pct >= 65) {
    return {
      text: 'text-emerald-400',
      bar: 'bg-emerald-400',
      border: 'border-emerald-400/20',
      bg: 'bg-emerald-400/10',
    };
  }
  if (pct >= 31) {
    return {
      text: 'text-amber-400',
      bar: 'bg-amber-400',
      border: 'border-amber-400/20',
      bg: 'bg-amber-400/10',
    };
  }
  return {
    text: 'text-red-400',
    bar: 'bg-red-400',
    border: 'border-red-400/20',
    bg: 'bg-red-400/10',
  };
}
