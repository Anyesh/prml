import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { kalmanFilter, kalmanSmoother, type LdsParams } from './kalman.js';
import type { Mat, Vec } from '../types.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('kalman_smoother');
const c = fixture.cases[0]!;

function expectVecClose(actual: Vec, expected: Vec, digits = 9) {
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, digits));
}

function expectMatClose(actual: Mat, expected: Mat, digits = 9) {
  actual.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, digits)));
}

describe('kalmanSmoother', () => {
  it('matches the plain-numpy RTS recursion (PRML 13.100-13.104)', () => {
    const observations = c['observations'] as Mat;
    const params = c['params'] as LdsParams;
    const expected = c['expected'] as { mean: Mat; cov: Mat[]; J: Mat[]; pairwiseCov: Mat[] };

    const filtered = kalmanFilter(observations, params);
    const smoothed = kalmanSmoother(filtered, params);

    smoothed.mean.forEach((m, n) => expectVecClose(m, expected.mean[n]!));
    smoothed.cov.forEach((v, n) => expectMatClose(v, expected.cov[n]!));
    smoothed.J.forEach((j, n) => expectMatClose(j, expected.J[n]!));
    smoothed.pairwiseCov.forEach((p, n) => expectMatClose(p, expected.pairwiseCov[n]!));
  });

  it('the smoothed estimate at the last time step equals the filtered estimate there', () => {
    const observations = c['observations'] as Mat;
    const params = c['params'] as LdsParams;
    const filtered = kalmanFilter(observations, params);
    const smoothed = kalmanSmoother(filtered, params);
    const last = filtered.length - 1;
    expectVecClose(smoothed.mean[last]!, filtered[last]!.mean);
    expectMatClose(smoothed.cov[last]!, filtered[last]!.cov);
  });

  it('smoothing never increases uncertainty: every smoothed variance is at most the filtered one', () => {
    const observations = c['observations'] as Mat;
    const params = c['params'] as LdsParams;
    const filtered = kalmanFilter(observations, params);
    const smoothed = kalmanSmoother(filtered, params);
    smoothed.cov.forEach((cov, n) => {
      cov.forEach((row, i) => {
        expect(row[i]!).toBeLessThanOrEqual(filtered[n]!.cov[i]![i]! + 1e-9);
      });
    });
  });
});
