import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { hmmEStep, hmmMStep, type HmmGaussianParams } from './em.js';
import type { Mat } from '../types.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('hmm_em');
function caseFor(fn: string): Case {
  const c = fixture.cases.find((c) => c.fn === fn);
  if (!c) throw new Error(`no case for ${fn}`);
  return c;
}

function expectMatClose(actual: Mat, expected: Mat, digits = 9) {
  actual.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, digits)));
}

function expectParamsClose(actual: HmmGaussianParams, expected: HmmGaussianParams) {
  actual.pi.forEach((v, k) => expect(v).toBeCloseTo(expected.pi[k]!, 9));
  expectMatClose(actual.A, expected.A);
  actual.components.forEach((c, k) => {
    const e = expected.components[k]!;
    c.mean.forEach((v, i) => expect(v).toBeCloseTo(e.mean[i]!, 9));
    c.cov.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(e.cov[i]![j]!, 9)));
  });
}

describe('hmmEStep', () => {
  it('matches the scaled forward-backward gamma, xi, and log-likelihood', () => {
    const c = caseFor('hmmEStep');
    const data = c['data'] as Mat;
    const params = c['params'] as HmmGaussianParams;
    const expected = c['expected'] as { gamma: Mat; xi: Mat[]; logLikelihood: number };
    const out = hmmEStep(data, params);
    expectMatClose(out.gamma, expected.gamma);
    out.xi.forEach((mat, n) => expectMatClose(mat, expected.xi[n]!));
    expect(out.logLikelihood).toBeCloseTo(expected.logLikelihood, 9);
  });
});

describe('hmmMStep', () => {
  it('matches the closed-form pi, A, mean, covariance update (PRML 13.18-13.21)', () => {
    const c = caseFor('hmmMStep');
    const data = c['data'] as Mat;
    const gamma = c['gamma'] as Mat;
    const xi = c['xi'] as Mat[];
    const expected = c['expected'] as HmmGaussianParams;
    const out = hmmMStep(data, gamma, xi);
    expectParamsClose(out, expected);
  });
});
