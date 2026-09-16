import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  vbGmmExpectedLogDet,
  vbGmmExpectedLogPi,
  vbGmmLowerBound,
  vbGmmMStep,
  vbGmmPredictiveLogPdf,
  vbGmmResponsibilities,
  type VbGmmPosterior,
  type VbGmmPrior,
} from './vbgmm.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('vbgmm');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('vbGmmExpectedLogPi / vbGmmExpectedLogDet', () => {
  it('matches the Dirichlet and Wishart expectations (PRML 10.65-10.66)', () => {
    const piCase = casesFor('vbGmmExpectedLogPi')[0]!;
    const posterior = piCase['posterior'] as VbGmmPosterior;
    (vbGmmExpectedLogPi(posterior)).forEach((v, k) => expect(v).toBeCloseTo((piCase['expected'] as number[])[k]!, 9));

    const detCase = casesFor('vbGmmExpectedLogDet')[0]!;
    const detPosterior = detCase['posterior'] as VbGmmPosterior;
    (vbGmmExpectedLogDet(detPosterior)).forEach((v, k) => expect(v).toBeCloseTo((detCase['expected'] as number[])[k]!, 9));
  });
});

describe('vbGmmResponsibilities', () => {
  it('matches the closed-form variational E-step (PRML 10.46-10.49, 10.67)', () => {
    for (const c of casesFor('vbGmmResponsibilities')) {
      const data = c['data'] as number[][];
      const posterior = c['posterior'] as VbGmmPosterior;
      const expected = c['expected'] as number[][];
      const out = vbGmmResponsibilities(data, posterior);
      out.forEach((row, n) => row.forEach((v, k) => expect(v).toBeCloseTo(expected[n]![k]!, 9)));
    }
  });

  it('sums to one at every point', () => {
    const c = casesFor('vbGmmResponsibilities')[0]!;
    const data = c['data'] as number[][];
    const posterior = c['posterior'] as VbGmmPosterior;
    const out = vbGmmResponsibilities(data, posterior);
    out.forEach((row) => expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 9));
  });
});

describe('vbGmmMStep', () => {
  it('matches the Dirichlet / Gaussian-Wishart update (PRML 10.58, 10.60-10.63)', () => {
    const c = casesFor('vbGmmMStep')[0]!;
    const data = c['data'] as number[][];
    const r = c['r'] as number[][];
    const prior = c['prior'] as VbGmmPrior;
    const expected = c['expected'] as VbGmmPosterior;
    const out = vbGmmMStep(data, r, prior);

    out.alpha.forEach((a, k) => expect(a).toBeCloseTo(expected.alpha[k]!, 9));
    out.components.forEach((comp, k) => {
      const e = expected.components[k]!;
      expect(comp.beta).toBeCloseTo(e.beta, 9);
      comp.mean.forEach((v, i) => expect(v).toBeCloseTo(e.mean[i]!, 9));
      comp.scale.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(e.scale[i]![j]!, 9)));
      expect(comp.dof).toBeCloseTo(e.dof, 9);
    });
  });
});

describe('vbGmmLowerBound', () => {
  it('matches the seven-term decomposition (PRML 10.70-10.77)', () => {
    const c = casesFor('vbGmmLowerBound')[0]!;
    const data = c['data'] as number[][];
    const r = c['r'] as number[][];
    const posterior = c['posterior'] as VbGmmPosterior;
    const prior = c['prior'] as VbGmmPrior;
    expect(vbGmmLowerBound(data, r, posterior, prior)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('vbGmmPredictiveLogPdf', () => {
  it('matches a mixture of Student-t densities (PRML 10.81-10.82)', () => {
    const c = casesFor('vbGmmPredictiveLogPdf')[0]!;
    const points = c['points'] as number[][];
    const posterior = c['posterior'] as VbGmmPosterior;
    const expected = c['expected'] as number[];
    points.forEach((x, i) => expect(vbGmmPredictiveLogPdf(x, posterior)).toBeCloseTo(expected[i]!, 9));
  });
});
