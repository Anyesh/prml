import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { weightedLogisticFit } from './weightedLogistic.js';

interface Fixture {
  readonly cases: readonly {
    readonly fn: string;
    readonly design: number[][];
    readonly targets: number[];
    readonly weights: number[];
    readonly expected: number[];
  }[];
}

const fixture = loadFixture<Fixture>('weightedLogistic');
const TOL = 1e-9;

describe('weightedLogisticFit', () => {
  for (const [i, c] of fixture.cases.entries()) {
    it(`converges to the same stationary point as an independent Newton solve, case ${i}`, () => {
      const fit = weightedLogisticFit(c.design, c.targets, c.weights);
      expect(fit.converged).toBe(true);
      expect(fit.weights).toHaveLength(c.expected.length);
      c.expected.forEach((e, j) => {
        expect(Math.abs(fit.weights[j]! - e)).toBeLessThanOrEqual(TOL * Math.max(1, Math.abs(e)));
      });
    });
  }
});
