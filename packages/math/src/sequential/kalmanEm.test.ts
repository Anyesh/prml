import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { kalmanEmExpectations, kalmanMStep, type KalmanSmoothResult, type LdsParams } from './kalman.js';
import type { Mat, Vec } from '../types.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('kalman_em');
function caseFor(fn: string): Case {
  const c = fixture.cases.find((c) => c.fn === fn);
  if (!c) throw new Error(`no case for ${fn}`);
  return c;
}

function expectVecClose(actual: Vec, expected: Vec, digits = 9) {
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, digits));
}

function expectMatClose(actual: Mat, expected: Mat, digits = 9) {
  actual.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, digits)));
}

describe('kalmanEmExpectations', () => {
  it('matches E[zn], E[zn zn^T], E[zn z(n-1)^T] from smoother output (PRML 13.105-13.107)', () => {
    const c = caseFor('kalmanEmExpectations');
    const smoothed: KalmanSmoothResult = {
      mean: c['smoothedMean'] as Vec[],
      cov: c['smoothedCov'] as Mat[],
      J: [],
      pairwiseCov: c['pairwiseCov'] as Mat[],
    };
    const expected = c['expected'] as { Ez: Mat; Ezz: Mat[]; Ezzlag: Mat[] };
    const out = kalmanEmExpectations(smoothed);
    out.Ez.forEach((v, n) => expectVecClose(v, expected.Ez[n]!));
    out.Ezz.forEach((m, n) => expectMatClose(m, expected.Ezz[n]!));
    out.Ezzlag.forEach((m, n) => expectMatClose(m, expected.Ezzlag[n]!));
  });
});

describe('kalmanMStep', () => {
  it('matches the closed-form parameter update (PRML 13.110-13.116)', () => {
    const c = caseFor('kalmanMStep');
    const observations = c['observations'] as Mat;
    const expectations = { Ez: c['Ez'] as Vec[], Ezz: c['Ezz'] as Mat[], Ezzlag: c['Ezzlag'] as Mat[] };
    const expected = c['expected'] as LdsParams;
    const out = kalmanMStep(observations, expectations);
    expectMatClose(out.A, expected.A);
    expectMatClose(out.Gamma, expected.Gamma);
    expectMatClose(out.C, expected.C);
    expectMatClose(out.Sigma, expected.Sigma);
    expectVecClose(out.mu0, expected.mu0);
    expectMatClose(out.V0, expected.V0);
  });
});
