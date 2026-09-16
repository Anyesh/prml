import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { adaBoostFit, adaBoostFunctionValue, adaBoostPredict, type AdaBoostFit } from './adaboost.js';

interface RoundExpected {
  readonly weights: readonly number[];
  readonly stump: { readonly featureIndex: number; readonly threshold: number; readonly polarity: 1 | -1 };
  readonly epsilon: number;
  readonly alpha: number;
}

interface RoundsCase {
  readonly fn: 'adaBoostRounds';
  readonly X: number[][];
  readonly targets: number[];
  readonly rounds: number;
  readonly expected: readonly RoundExpected[];
}

interface EvalCase {
  readonly fn: 'adaBoostFunctionValue' | 'adaBoostPredict';
  readonly x: number[];
  readonly upToRound: number | null;
  readonly expected: number;
}

interface Fixture {
  readonly dataset: { readonly X: number[][]; readonly targets: number[] };
  readonly rounds: number;
  readonly cases: readonly (RoundsCase | EvalCase)[];
}

const fixture = loadFixture<Fixture>('adaboost');
const TOL = 1e-9;

function closeTo(actual: number, expected: number): boolean {
  return Math.abs(actual - expected) <= TOL * Math.max(1, Math.abs(expected));
}

const roundsCase = fixture.cases.find((c): c is RoundsCase => c.fn === 'adaBoostRounds')!;
const fit: AdaBoostFit = adaBoostFit(roundsCase.X, roundsCase.targets, roundsCase.rounds);

describe('adaBoostFit', () => {
  it('runs exactly the requested number of rounds', () => {
    expect(fit.rounds).toHaveLength(roundsCase.rounds);
  });

  roundsCase.expected.forEach((expectedRound, m) => {
    it(`round ${m} matches the independent numpy trace`, () => {
      const round = fit.rounds[m]!;
      expect(round.weights).toHaveLength(expectedRound.weights.length);
      expectedRound.weights.forEach((w, i) => {
        expect(closeTo(round.weights[i]!, w)).toBe(true);
      });
      expect(round.stump.featureIndex).toBe(expectedRound.stump.featureIndex);
      expect(round.stump.polarity).toBe(expectedRound.stump.polarity);
      expect(closeTo(round.stump.threshold, expectedRound.stump.threshold)).toBe(true);
      expect(closeTo(round.epsilon, expectedRound.epsilon)).toBe(true);
      expect(closeTo(round.alpha, expectedRound.alpha)).toBe(true);
    });
  });
});

describe('adaBoostFunctionValue / adaBoostPredict', () => {
  for (const c of fixture.cases) {
    if (c.fn !== 'adaBoostFunctionValue' && c.fn !== 'adaBoostPredict') continue;
    const upTo = c.upToRound ?? undefined;
    it(`${c.fn} at x = [${c.x}], upToRound = ${c.upToRound}`, () => {
      if (c.fn === 'adaBoostFunctionValue') {
        const actual = adaBoostFunctionValue(fit, c.x, upTo);
        expect(closeTo(actual, c.expected)).toBe(true);
      } else {
        expect(adaBoostPredict(fit, c.x, upTo)).toBe(c.expected);
      }
    });
  }
});

describe('epsilon clamp guard', () => {
  it('keeps alpha finite when the first stump is a perfect classifier', () => {
    const X = [[0], [1], [2], [3]];
    const targets = [-1, -1, 1, 1];
    const fitResult = adaBoostFit(X, targets, 1);
    const round = fitResult.rounds[0]!;
    expect(round.epsilon).toBe(0);
    expect(Number.isFinite(round.alpha)).toBe(true);
    expect(closeTo(round.alpha, Math.log((1 - 1e-10) / 1e-10))).toBe(true);
  });
});
