import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import type { Mat, Vec } from '../types.js';
import { faEmStep, faMarginalCov, type FaParams } from './factorAnalysis.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('factorAnalysis');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('faMarginalCov', () => {
  it('matches W W^T + diag(Psi), anisotropic unlike PPCA', () => {
    const c = casesFor('faMarginalCov')[0]!;
    const result = faMarginalCov(c['w'] as Mat, c['psi'] as Vec);
    const expected = c['expected'] as number[][];
    result.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('faEmStep', () => {
  it('reproduces a hand-rolled E-step-then-M-step iterate trace', () => {
    const c = casesFor('faEmTrace')[0]!;
    const data = c['data'] as Mat;
    const mean = c['mean'] as Vec;
    let params: FaParams = { mean, w: c['initialW'] as Mat, psi: c['initialPsi'] as Vec };
    const expectedTrace = c['expectedTrace'] as Array<{
      ez: number[][];
      wAfterMStep: number[][];
      psiAfterMStep: number[];
    }>;

    for (const expected of expectedTrace) {
      const step = faEmStep(data, params);
      step.ez.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected.ez[i]![j]!, 9)));
      step.params.w.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected.wAfterMStep[i]![j]!, 9)));
      step.params.psi.forEach((v, i) => expect(v).toBeCloseTo(expected.psiAfterMStep[i]!, 9));
      params = step.params;
    }
  });
});
