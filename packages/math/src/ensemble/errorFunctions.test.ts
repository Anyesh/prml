import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  absoluteResidualError,
  crossEntropyMarginError,
  exponentialMarginError,
  hingeMarginError,
  misclassificationMarginError,
  squaredResidualError,
} from './errorFunctions.js';

type MarginCase = { readonly fn: string; readonly z: number; readonly expected: number };
type ResidualCase = { readonly fn: string; readonly residual: number; readonly expected: number };

interface Fixture {
  readonly cases: readonly (MarginCase | ResidualCase)[];
}

const fixture = loadFixture<Fixture>('errorFunctions');
const TOL = 1e-9;

function closeTo(actual: number, expected: number): boolean {
  return Math.abs(actual - expected) <= TOL * Math.max(1, Math.abs(expected));
}

const MARGIN_FNS: Record<string, (z: number) => number> = {
  exponentialMarginError,
  crossEntropyMarginError,
  hingeMarginError,
  misclassificationMarginError,
};

const RESIDUAL_FNS: Record<string, (r: number) => number> = {
  squaredResidualError,
  absoluteResidualError,
};

describe('margin error functions', () => {
  for (const c of fixture.cases) {
    if (!(c.fn in MARGIN_FNS)) continue;
    const margin = c as MarginCase;
    it(`${margin.fn} matches scipy at z = ${margin.z}`, () => {
      const actual = MARGIN_FNS[margin.fn]!(margin.z);
      expect(closeTo(actual, margin.expected)).toBe(true);
    });
  }
});

describe('residual error functions', () => {
  for (const c of fixture.cases) {
    if (!(c.fn in RESIDUAL_FNS)) continue;
    const residual = c as ResidualCase;
    it(`${residual.fn} matches the reference at residual = ${residual.residual}`, () => {
      const actual = RESIDUAL_FNS[residual.fn]!(residual.residual);
      expect(closeTo(actual, residual.expected)).toBe(true);
    });
  }
});

describe('numerical stability', () => {
  it('crossEntropyMarginError stays finite for a very negative margin', () => {
    expect(Number.isFinite(crossEntropyMarginError(-700))).toBe(true);
  });

  it('exponentialMarginError does not overflow for a very positive margin', () => {
    expect(exponentialMarginError(50)).toBeGreaterThan(0);
    expect(Number.isFinite(exponentialMarginError(50))).toBe(true);
  });
});
