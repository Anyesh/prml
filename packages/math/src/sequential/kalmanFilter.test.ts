import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { kalmanFilter, kalmanLogLikelihood, type LdsParams } from './kalman.js';
import type { Mat, Vec } from '../types.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('kalman_filter');
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

describe('kalmanFilter', () => {
  it('matches the plain-numpy filter recursion (PRML 13.89-13.97) at every step', () => {
    const c = caseFor('kalmanFilter');
    const observations = c['observations'] as Mat;
    const params = c['params'] as LdsParams;
    const expected = c['expected'] as {
      mean: Mat;
      cov: Mat[];
      predictedMean: Mat;
      predictedCov: Mat[];
      gain: Mat[];
      c: Vec;
      logC: Vec;
    };

    const steps = kalmanFilter(observations, params);
    steps.forEach((step, n) => {
      expectVecClose(step.mean, expected.mean[n]!);
      expectMatClose(step.cov, expected.cov[n]!);
      expectVecClose(step.predictedMean, expected.predictedMean[n]!);
      expectMatClose(step.predictedCov, expected.predictedCov[n]!);
      expectMatClose(step.gain, expected.gain[n]!);
      expect(step.c).toBeCloseTo(expected.c[n]!, 9);
      expect(step.logC).toBeCloseTo(expected.logC[n]!, 9);
    });
  });
});

describe('kalmanLogLikelihood', () => {
  it('matches sum(logC) (PRML 13.63 analogue)', () => {
    const c = caseFor('kalmanLogLikelihood');
    const logC = c['logC'] as Vec;
    expect(kalmanLogLikelihood(logC)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('the noiseless limit (Exercise 13.27)', () => {
  it('as observation noise goes to zero, the filtered mean converges to the current observation', () => {
    const params: LdsParams = {
      A: [[1]],
      Gamma: [[0.2]],
      C: [[1]],
      Sigma: [[1e-10]],
      mu0: [0],
      V0: [[5]],
    };
    const observations: Mat = [[3.7], [1.2], [-0.4]];
    const steps = kalmanFilter(observations, params);
    steps.forEach((step, n) => {
      expect(step.mean[0]!).toBeCloseTo(observations[n]![0]!, 4);
      expect(step.cov[0]![0]!).toBeLessThan(1e-6);
    });
  });
});
