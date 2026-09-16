import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { gmmLogLikelihood, type GmmParams } from './gmm.js';
import { emKlGap, emLowerBound, expectedCompleteDataLogLikelihood, gmmEmStep, responsibilityEntropy } from './em.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('em');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

function expectParamsClose(actual: GmmParams, expected: GmmParams, digits = 9) {
  expect(actual.components).toHaveLength(expected.components.length);
  actual.components.forEach((c, k) => {
    const e = expected.components[k]!;
    expect(c.weight).toBeCloseTo(e.weight, digits);
    c.mean.forEach((v, i) => expect(v).toBeCloseTo(e.mean[i]!, digits));
    c.cov.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(e.cov[i]![j]!, digits)));
  });
}

describe('expectedCompleteDataLogLikelihood', () => {
  it('matches the Q function (PRML 9.40)', () => {
    const c = casesFor('expectedCompleteDataLogLikelihood')[0]!;
    const data = c['data'] as number[][];
    const responsibilities = c['responsibilities'] as number[][];
    const params = c['params'] as GmmParams;
    expect(expectedCompleteDataLogLikelihood(data, responsibilities, params)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('responsibilityEntropy', () => {
  it('matches -sum r log r, treating 0 log 0 as 0', () => {
    const c = casesFor('responsibilityEntropy')[0]!;
    const responsibilities = c['responsibilities'] as number[][];
    expect(responsibilityEntropy(responsibilities)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('emLowerBound', () => {
  it('equals the expected complete-data log-likelihood plus responsibility entropy', () => {
    const c = casesFor('emLowerBound_atExactPosterior')[0]!;
    const data = c['data'] as number[][];
    const responsibilities = c['responsibilities'] as number[][];
    const params = c['params'] as GmmParams;
    expect(emLowerBound(data, responsibilities, params)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('emKlGap', () => {
  it('is (numerically) zero when the responsibilities are the exact posterior for these parameters', () => {
    const c = casesFor('emKlGap_atExactPosterior')[0]!;
    const data = c['data'] as number[][];
    const responsibilities = c['responsibilities'] as number[][];
    const params = c['params'] as GmmParams;
    expect(gmmLogLikelihood(data, params)).toBeCloseTo(c['expectedLogLikelihood'] as number, 9);
    const gap = emKlGap(data, responsibilities, params);
    expect(gap).toBeCloseTo(c['expectedGap'] as number, 6);
    expect(gap).toBeGreaterThanOrEqual(-1e-9);
  });

  it('is strictly positive once q is perturbed away from the exact posterior', () => {
    const c = casesFor('emKlGap_awayFromPosterior')[0]!;
    const data = c['data'] as number[][];
    const responsibilities = c['responsibilities'] as number[][];
    const params = c['params'] as GmmParams;
    const gap = emKlGap(data, responsibilities, params);
    expect(gap).toBeCloseTo(c['expectedGap'] as number, 6);
    expect(gap).toBeGreaterThan(1e-6);
  });
});

describe('gmmEmStep', () => {
  it('reproduces a hand-rolled E-step-then-M-step iterate trace', () => {
    const c = casesFor('gmmEmTrace')[0]!;
    const data = c['data'] as number[][];
    let params = c['initialParams'] as GmmParams;
    const expectedTrace = c['expectedTrace'] as Array<{
      responsibilities: number[][];
      logLikelihoodBeforeMStep: number;
      paramsAfterMStep: GmmParams;
    }>;

    for (const expected of expectedTrace) {
      const step = gmmEmStep(data, params);
      step.responsibilities.forEach((row, n) => row.forEach((v, k) => expect(v).toBeCloseTo(expected.responsibilities[n]![k]!, 9)));
      expect(step.logLikelihood).toBeCloseTo(expected.logLikelihoodBeforeMStep, 9);
      expectParamsClose(step.params, expected.paramsAfterMStep);
      params = step.params;
    }
  });

  it('never decreases the log-likelihood from one iterate to the next', () => {
    const c = casesFor('gmmEmTrace')[0]!;
    const data = c['data'] as number[][];
    let params = c['initialParams'] as GmmParams;
    let previous = -Infinity;
    for (let i = 0; i < 4; i++) {
      const step = gmmEmStep(data, params);
      expect(step.logLikelihood).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = step.logLikelihood;
      params = step.params;
    }
  });
});
