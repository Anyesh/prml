import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { gmmEStep, gmmInit, gmmLogLikelihood, gmmLogPdf, gmmMStep, gmmPdf, gmmResponsibilities, type GmmParams } from './gmm.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('gmm');
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

describe('gmmLogPdf / gmmPdf', () => {
  it('matches a scipy-backed mixture density at several points (PRML 9.7)', () => {
    for (const c of casesFor('gmmLogPdf')) {
      const x = c['x'] as number[];
      const params = c['params'] as GmmParams;
      expect(gmmLogPdf(x, params)).toBeCloseTo(c['expected'] as number, 9);
    }
  });

  it('pdf is exp(logPdf)', () => {
    const c = casesFor('gmmLogPdf')[0]!;
    const x = c['x'] as number[];
    const params = c['params'] as GmmParams;
    expect(gmmPdf(x, params)).toBeCloseTo(Math.exp(gmmLogPdf(x, params)), 9);
  });
});

describe('gmmResponsibilities', () => {
  it('matches softmax-weighted component posteriors (PRML 9.13)', () => {
    for (const c of casesFor('gmmResponsibilities')) {
      const x = c['x'] as number[];
      const params = c['params'] as GmmParams;
      const expected = c['expected'] as number[];
      const out = gmmResponsibilities(x, params);
      out.forEach((v, k) => expect(v).toBeCloseTo(expected[k]!, 9));
    }
  });

  it('sums to one at every point', () => {
    for (const c of casesFor('gmmResponsibilities')) {
      const x = c['x'] as number[];
      const params = c['params'] as GmmParams;
      const out = gmmResponsibilities(x, params);
      expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    }
  });
});

describe('gmmEStep', () => {
  it('matches per-point responsibilities stacked into a matrix', () => {
    const c = casesFor('gmmEStep')[0]!;
    const data = c['data'] as number[][];
    const params = c['params'] as GmmParams;
    const expected = c['expected'] as number[][];
    const out = gmmEStep(data, params);
    out.forEach((row, n) => row.forEach((v, k) => expect(v).toBeCloseTo(expected[n]![k]!, 9)));
  });
});

describe('gmmLogLikelihood', () => {
  it('matches the sum of per-point log mixture densities (PRML 9.14)', () => {
    const c = casesFor('gmmLogLikelihood')[0]!;
    const data = c['data'] as number[][];
    const params = c['params'] as GmmParams;
    expect(gmmLogLikelihood(data, params)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('gmmInit', () => {
  it('builds an initial mixture from a hard clustering: cluster fraction, mean, empirical covariance', () => {
    const c = casesFor('gmmInit')[0]!;
    const data = c['data'] as number[][];
    const means = c['means'] as number[][];
    const assignments = c['assignments'] as number[];
    const k = c['k'] as number;
    const expected = c['expected'] as GmmParams;
    expectParamsClose(gmmInit(data, means, assignments, k), expected);
  });
});

describe('gmmMStep', () => {
  it('matches the closed-form responsibility-weighted update (PRML 9.17, 9.19, 9.22)', () => {
    const c = casesFor('gmmMStep')[0]!;
    const data = c['data'] as number[][];
    const responsibilities = c['responsibilities'] as number[][];
    const expected = c['expected'] as GmmParams;
    expectParamsClose(gmmMStep(data, responsibilities), expected);
  });
});
