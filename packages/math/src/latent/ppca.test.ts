import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import type { Mat, Vec } from '../types.js';
import { ppcaEmStep, ppcaLatentPosterior, ppcaLogLikelihood, ppcaMarginalLogPdf, ppcaMLE, type PpcaParams } from './ppca.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('ppca');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

function expectMatClose(actual: Mat, expected: number[][], digits = 9) {
  actual.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, digits)));
}

function expectVecClose(actual: Vec, expected: number[], digits = 9) {
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, digits));
}

describe('ppcaMarginalLogPdf', () => {
  it('matches N(x | mean, W W^T + sigma^2 I) (PRML 12.31-12.36)', () => {
    const c = casesFor('ppcaMarginalLogPdf')[0]!;
    const params: PpcaParams = { mean: c['mean'] as Vec, w: c['w'] as Mat, sigma2: c['sigma2'] as number };
    const result = ppcaMarginalLogPdf(c['x'] as Vec, params);
    expect(result).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('ppcaLatentPosterior', () => {
  it('matches PRML 12.42: mean depends on x, covariance does not', () => {
    const c = casesFor('ppcaLatentPosterior')[0]!;
    const params: PpcaParams = { mean: c['mean'] as Vec, w: c['w'] as Mat, sigma2: c['sigma2'] as number };
    const expected = c['expected'] as { mean: number[]; cov: number[][] };
    const result = ppcaLatentPosterior(c['x'] as Vec, params);
    expectVecClose(result.mean, expected.mean);
    expectMatClose(result.cov, expected.cov);
  });
});

describe('ppcaMLE', () => {
  it('reproduces the closed-form solution (PRML 12.45-12.46) with R fixed to the identity', () => {
    const c = casesFor('ppcaMLE')[0]!;
    const data = c['data'] as Mat;
    const latentDim = c['latentDim'] as number;
    const expected = c['expected'] as { mean: number[]; w: number[][]; sigma2: number };
    const result = ppcaMLE(data, latentDim);
    expectVecClose(result.mean, expected.mean);
    expect(result.sigma2).toBeCloseTo(expected.sigma2, 9);
    expectMatClose(result.w, expected.w);
  });

  it('has a marginal log-likelihood that never exceeds the unconstrained Gaussian fit (M = D)', () => {
    const c = casesFor('ppcaMLE')[0]!;
    const data = c['data'] as Mat;
    const dim = (c['expected'] as { mean: number[] }).mean.length;
    const restricted = ppcaMLE(data, 1);
    const full = ppcaMLE(data, dim);
    const llRestricted = ppcaLogLikelihood(data, restricted);
    const llFull = ppcaLogLikelihood(data, full);
    expect(llFull).toBeGreaterThanOrEqual(llRestricted - 1e-6);
  });
});

describe('ppcaEmStep', () => {
  it('reproduces a hand-rolled E-step-then-M-step iterate trace (PRML 12.54-12.57)', () => {
    const c = casesFor('ppcaEmTrace')[0]!;
    const data = c['data'] as Mat;
    const mean = c['mean'] as Vec;
    let params: PpcaParams = { mean, w: c['initialW'] as Mat, sigma2: c['initialSigma2'] as number };
    const expectedTrace = c['expectedTrace'] as Array<{
      ez: number[][];
      logLikelihoodBeforeMStep: number;
      wAfterMStep: number[][];
      sigma2AfterMStep: number;
    }>;

    for (const expected of expectedTrace) {
      const step = ppcaEmStep(data, params);
      expectMatClose(step.ez, expected.ez);
      expect(step.logLikelihood).toBeCloseTo(expected.logLikelihoodBeforeMStep, 9);
      expectMatClose(step.params.w, expected.wAfterMStep);
      expect(step.params.sigma2).toBeCloseTo(expected.sigma2AfterMStep, 9);
      params = step.params;
    }
  });

  it('never decreases the marginal log-likelihood from one iterate to the next', () => {
    const c = casesFor('ppcaEmTrace')[0]!;
    const data = c['data'] as Mat;
    const mean = c['mean'] as Vec;
    let params: PpcaParams = { mean, w: c['initialW'] as Mat, sigma2: c['initialSigma2'] as number };
    let previous = -Infinity;
    for (let i = 0; i < 4; i++) {
      const step = ppcaEmStep(data, params);
      expect(step.logLikelihood).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = step.logLikelihood;
      params = step.params;
    }
  });
});
