import { describe, expect, it } from 'vitest';

import { buildScale } from './Plot.js';
import type { Domain, ScaleKind } from './frame.js';

const scale = (kind: ScaleKind, domain: Domain) => buildScale(kind, domain, [0, 400], 'x');

describe('Plot scale kinds', () => {
  it('rejects a log domain that reaches zero, rather than drawing NaNs', () => {
    expect(() => scale('log', [0, 100])).toThrow(/strictly positive/);
    expect(() => scale('log', [-1, 100])).toThrow(/strictly positive/);
  });

  it('places a decade at the same spacing anywhere on a log axis', () => {
    const logScale = scale('log', [1, 1000]);
    const firstDecade = logScale(10) - logScale(1);
    const lastDecade = logScale(1000) - logScale(100);
    expect(firstDecade).toBeCloseTo(lastDecade, 10);
  });
});
