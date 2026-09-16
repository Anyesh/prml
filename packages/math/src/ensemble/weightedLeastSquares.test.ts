import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { weightedLeastSquaresFit } from './weightedLeastSquares.js';

interface Fixture {
  readonly cases: readonly {
    readonly fn: string;
    readonly design: number[][];
    readonly targets: number[];
    readonly weights: number[];
    readonly expected: number[];
  }[];
}

const fixture = loadFixture<Fixture>('weightedLeastSquares');
const TOL = 1e-9;

describe('weightedLeastSquaresFit', () => {
  for (const [i, c] of fixture.cases.entries()) {
    it(`matches the weighted normal equations, case ${i}`, () => {
      const fit = weightedLeastSquaresFit(c.design, c.targets, c.weights);
      expect(fit).toHaveLength(c.expected.length);
      c.expected.forEach((e, j) => {
        expect(Math.abs(fit[j]! - e)).toBeLessThanOrEqual(TOL * Math.max(1, Math.abs(e)));
      });
    });
  }
});
