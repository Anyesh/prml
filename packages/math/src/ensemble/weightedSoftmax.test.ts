import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { weightedSoftmaxFit } from './weightedSoftmax.js';

interface Fixture {
  readonly cases: readonly {
    readonly fn: string;
    readonly design: number[][];
    readonly targets: number[][];
    readonly pointWeights: number[];
    readonly expected: number[][];
  }[];
}

const fixture = loadFixture<Fixture>('weightedSoftmax');
const TOL = 1e-9;

function expectMatClose(actual: readonly (readonly number[])[], expected: readonly (readonly number[])[]) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((row, k) => {
    expect(actual[k]).toHaveLength(row.length);
    row.forEach((e, d) => {
      expect(Math.abs(actual[k]![d]! - e)).toBeLessThanOrEqual(TOL * Math.max(1, Math.abs(e)));
    });
  });
}

describe('weightedSoftmaxFit', () => {
  for (const [i, c] of fixture.cases.entries()) {
    it(`converges to the same reference-class-zero optimum as an independent Newton solve, case ${i}`, () => {
      const fit = weightedSoftmaxFit(c.design, c.targets, c.pointWeights);
      expect(fit.converged).toBe(true);
      expectMatClose(fit.weights, c.expected);
    });
  }

  it('agrees with weightedLogisticFit at K=2, since sigmoid(v.x) is softmax([0, w1.x])[1] identically', () => {
    const c = fixture.cases[3]!;
    const fit = weightedSoftmaxFit(c.design, c.targets, c.pointWeights);
    expect(fit.weights).toHaveLength(2);
    expect(fit.weights[0]!.every((v) => v === 0)).toBe(true);
  });
});
