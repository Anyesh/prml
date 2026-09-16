import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  vlrFit,
  vlrLowerBound,
  vlrQAlphaParams,
  vlrUpdate,
  type VlrPosterior,
  type VlrPrior,
} from './linearRegression.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('variationalLinearRegression');
function caseFor(fn: string): Case {
  const c = fixture.cases.find((c) => c.fn === fn);
  if (!c) throw new Error(`no fixture case for ${fn}`);
  return c;
}

function expectPosteriorClose(actual: VlrPosterior, expected: VlrPosterior, digits = 9) {
  expect(actual.a).toBeCloseTo(expected.a, digits);
  expect(actual.b).toBeCloseTo(expected.b, digits);
  actual.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, digits));
  actual.cov.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected.cov[i]![j]!, digits)));
}

describe('vlrQAlphaParams', () => {
  it('matches a_N = a0 + M/2, b_N = b0 + 0.5 E[w^Tw] (PRML 10.97-10.98)', () => {
    const c = caseFor('vlrQAlphaParams');
    const mean = c['mean'] as number[];
    const cov = c['cov'] as number[][];
    const expected = c['expected'] as { a: number; b: number };
    const out = vlrQAlphaParams(c['a0'] as number, c['b0'] as number, mean, cov);
    expect(out.a).toBeCloseTo(expected.a, 9);
    expect(out.b).toBeCloseTo(expected.b, 9);
  });
});

describe('vlrUpdate', () => {
  it('matches one closed-form coordinate-ascent sweep of q(w) and q(alpha) (PRML 10.94-10.99)', () => {
    const c = caseFor('vlrUpdate');
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const beta = c['beta'] as number;
    const prior = c['prior'] as VlrPrior;
    const current = c['current'] as { a: number; b: number };
    const expected = c['expected'] as VlrPosterior;
    const out = vlrUpdate(design, targets, beta, prior, current);
    expectPosteriorClose(out, expected);
  });
});

describe('vlrLowerBound', () => {
  it('matches the five-term evidence lower bound (PRML 10.107)', () => {
    const c = caseFor('vlrLowerBound');
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const beta = c['beta'] as number;
    const prior = c['prior'] as VlrPrior;
    const posterior = c['posterior'] as VlrPosterior;
    expect(vlrLowerBound(design, targets, beta, prior, posterior)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('vlrFit', () => {
  it('reproduces an independently computed multi-iteration trace', () => {
    const c = caseFor('vlrFit');
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const beta = c['beta'] as number;
    const prior = c['prior'] as VlrPrior;
    const iters = c['iters'] as number;
    const expectedTrace = c['expectedTrace'] as VlrPosterior[];
    const expectedLowerBoundTrace = c['expectedLowerBoundTrace'] as number[];

    const result = vlrFit(design, targets, beta, prior, iters);
    expect(result.posteriorHistory).toHaveLength(expectedTrace.length);
    result.posteriorHistory.forEach((p, i) => expectPosteriorClose(p, expectedTrace[i]!));
    result.lowerBoundHistory.forEach((lb, i) => expect(lb).toBeCloseTo(expectedLowerBoundTrace[i]!, 9));
  });

  it('the lower bound rises monotonically, as coordinate ascent on a bound must', () => {
    const c = caseFor('vlrFit');
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const beta = c['beta'] as number;
    const prior = c['prior'] as VlrPrior;
    const iters = c['iters'] as number;
    const result = vlrFit(design, targets, beta, prior, iters);
    for (let i = 1; i < result.lowerBoundHistory.length; i++) {
      expect(result.lowerBoundHistory[i]!).toBeGreaterThanOrEqual(result.lowerBoundHistory[i - 1]! - 1e-9);
    }
  });
});
