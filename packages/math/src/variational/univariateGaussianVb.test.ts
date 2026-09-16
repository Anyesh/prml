import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  univariateGaussianVbFit,
  type NormalGammaPrior,
  type UnivariateGaussianVbPosterior,
} from './univariateGaussianVb.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('univariateGaussianVb');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('univariateGaussianVbFit', () => {
  it('matches the coordinate-ascent trajectory at every sweep (PRML 10.25-10.30)', () => {
    for (const c of casesFor('univariateGaussianVbFit')) {
      const data = c['data'] as number[];
      const prior = c['prior'] as NormalGammaPrior;
      const initial = c['initial'] as UnivariateGaussianVbPosterior;
      const iterations = c['iterations'] as number;
      const expected = c['expected'] as UnivariateGaussianVbPosterior[];
      const trajectory = univariateGaussianVbFit(data, prior, initial, iterations);
      expect(trajectory).toHaveLength(expected.length);
      trajectory.forEach((step, i) => {
        expect(step.qMu.mu).toBeCloseTo(expected[i]!.qMu.mu, 9);
        expect(step.qMu.lambda).toBeCloseTo(expected[i]!.qMu.lambda, 9);
        expect(step.qTau.a).toBeCloseTo(expected[i]!.qTau.a, 9);
        expect(step.qTau.b).toBeCloseTo(expected[i]!.qTau.b, 9);
      });
    }
  });
});

describe('univariateGaussianVbFit convergence under a noninformative prior', () => {
  it('drives E[mu] to the sample mean and E[tau] to the reciprocal unbiased sample variance (PRML 10.31-10.33)', () => {
    const c = casesFor('univariateGaussianVbConvergence')[0]!;
    const data = c['data'] as number[];
    const prior = c['prior'] as NormalGammaPrior;
    const initial = c['initial'] as UnivariateGaussianVbPosterior;
    const iterations = c['iterations'] as number;
    const trajectory = univariateGaussianVbFit(data, prior, initial, iterations);
    const final = trajectory[trajectory.length - 1]!;
    expect(final.qMu.mu).toBeCloseTo(c['expectedMu'] as number, 9);
    expect(final.qTau.a / final.qTau.b).toBeCloseTo(c['expectedExpectedTau'] as number, 9);
  });
});
