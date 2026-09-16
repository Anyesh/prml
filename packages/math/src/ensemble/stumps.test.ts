import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { fitDecisionStump, stumpPredict } from './stumps.js';

interface StumpCase {
  readonly fn: 'fitDecisionStump';
  readonly name: string;
  readonly X: number[][];
  readonly targets: number[];
  readonly weights: number[];
  readonly expected: {
    readonly featureIndex: number;
    readonly threshold: number;
    readonly polarity: 1 | -1;
    readonly weightedError: number;
  };
}

interface Fixture {
  readonly cases: readonly StumpCase[];
}

const fixture = loadFixture<Fixture>('stumps');
const TOL = 1e-9;

function closeTo(actual: number, expected: number): boolean {
  return Math.abs(actual - expected) <= TOL * Math.max(1, Math.abs(expected));
}

describe('fitDecisionStump', () => {
  for (const c of fixture.cases) {
    it(`matches the independent numpy search: ${c.name}`, () => {
      const fit = fitDecisionStump(c.X, c.targets, c.weights);
      expect(fit.stump.featureIndex).toBe(c.expected.featureIndex);
      expect(fit.stump.polarity).toBe(c.expected.polarity);
      expect(closeTo(fit.stump.threshold, c.expected.threshold)).toBe(true);
      expect(closeTo(fit.weightedError, c.expected.weightedError)).toBe(true);
    });
  }
});

describe('stumpPredict', () => {
  it('predicts the polarity above the threshold and its negation at or below it', () => {
    const stump = { featureIndex: 1, threshold: 0.5, polarity: 1 as const };
    expect(stumpPredict(stump, [0, 1])).toBe(1);
    expect(stumpPredict(stump, [0, 0.5])).toBe(-1);
    expect(stumpPredict(stump, [0, 0])).toBe(-1);
  });

  it('flips with negative polarity', () => {
    const stump = { featureIndex: 0, threshold: 2, polarity: -1 as const };
    expect(stumpPredict(stump, [5, 0])).toBe(-1);
    expect(stumpPredict(stump, [1, 0])).toBe(1);
  });
});
