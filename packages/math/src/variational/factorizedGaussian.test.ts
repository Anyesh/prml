import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  factorizedGaussianForwardKlFit,
  factorizedGaussianReverseKl,
  type BivariateGaussianPrecision,
  type FactorizedGaussian,
} from './factorizedGaussian.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('factorizedGaussian');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

function expectFactorizedClose(actual: FactorizedGaussian, expected: FactorizedGaussian) {
  expect(actual.q1.mu).toBeCloseTo(expected.q1.mu, 9);
  expect(actual.q1.sigma2).toBeCloseTo(expected.q1.sigma2, 9);
  expect(actual.q2.mu).toBeCloseTo(expected.q2.mu, 9);
  expect(actual.q2.sigma2).toBeCloseTo(expected.q2.sigma2, 9);
}

describe('factorizedGaussianForwardKlFit', () => {
  it('matches the unrolled coordinate-ascent recursion at every step (PRML 10.12-10.15)', () => {
    for (const c of casesFor('factorizedGaussianForwardKlFit')) {
      const p = c['p'] as BivariateGaussianPrecision;
      const initial = c['initial'] as FactorizedGaussian;
      const iterations = c['iterations'] as number;
      const expected = c['expected'] as FactorizedGaussian[];
      const trajectory = factorizedGaussianForwardKlFit(p, initial, iterations);
      expect(trajectory).toHaveLength(expected.length);
      trajectory.forEach((step, i) => expectFactorizedClose(step, expected[i]!));
    }
  });

  it('the true mean with block precisions is a fixed point', () => {
    const cases = casesFor('factorizedGaussianForwardKlFit');
    const fixedCase = cases[cases.length - 1]!;
    const p = fixedCase['p'] as BivariateGaussianPrecision;
    const initial = fixedCase['initial'] as FactorizedGaussian;
    const [, after] = factorizedGaussianForwardKlFit(p, initial, 1);
    expect(after!.q1.mu).toBeCloseTo(initial.q1.mu, 9);
    expect(after!.q2.mu).toBeCloseTo(initial.q2.mu, 9);
  });
});

describe('factorizedGaussianReverseKl', () => {
  it('matches the true marginal mean and variance (PRML 10.17)', () => {
    for (const c of casesFor('factorizedGaussianReverseKl')) {
      const p = c['p'] as BivariateGaussianPrecision;
      const expected = c['expected'] as FactorizedGaussian;
      expectFactorizedClose(factorizedGaussianReverseKl(p), expected);
    }
  });

  it('reports a variance at least as large as the forward-KL fixed point (the under- vs over-coverage asymmetry)', () => {
    const c = casesFor('factorizedGaussianReverseKl')[0]!;
    const p = c['p'] as BivariateGaussianPrecision;
    const reverse = factorizedGaussianReverseKl(p);
    const trajectory = factorizedGaussianForwardKlFit(p, reverse, 10);
    const forward = trajectory[trajectory.length - 1]!;
    expect(reverse.q1.sigma2).toBeGreaterThanOrEqual(forward.q1.sigma2);
    expect(reverse.q2.sigma2).toBeGreaterThanOrEqual(forward.q2.sigma2);
  });
});
