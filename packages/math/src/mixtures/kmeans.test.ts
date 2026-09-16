import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { loadFixture } from '../testing/fixture.js';
import { kmeansAssign, kmeansDistortion, kmeansFit, kmeansInit, kmeansUpdateMeans } from './kmeans.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('kmeans');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('kmeansAssign', () => {
  it('assigns each point to its nearest mean, ties won by the lowest index', () => {
    const c = casesFor('kmeansAssign')[0]!;
    const data = c['data'] as number[][];
    const means = c['means'] as number[][];
    const expected = c['expected'] as number[];
    expect(kmeansAssign(data, means)).toEqual(expected);
  });
});

describe('kmeansDistortion', () => {
  it('matches sum of squared distances to the assigned mean (PRML 9.1)', () => {
    const c = casesFor('kmeansDistortion')[0]!;
    const data = c['data'] as number[][];
    const means = c['means'] as number[][];
    const assignments = c['assignments'] as number[];
    expect(kmeansDistortion(data, means, assignments)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('kmeansUpdateMeans', () => {
  it('updates each mean to the average of its assigned points (PRML 9.4)', () => {
    const c = casesFor('kmeansUpdateMeans')[0]!;
    const data = c['data'] as number[][];
    const assignments = c['assignments'] as number[];
    const k = c['k'] as number;
    const previousMeans = c['previousMeans'] as number[][];
    const expected = c['expected'] as number[][];
    const out = kmeansUpdateMeans(data, assignments, k, previousMeans);
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('leaves a cluster with no assigned points at its previous mean rather than producing NaN', () => {
    const c = casesFor('kmeansUpdateMeans_emptyCluster')[0]!;
    const data = c['data'] as number[][];
    const assignments = c['assignments'] as number[];
    const k = c['k'] as number;
    const previousMeans = c['previousMeans'] as number[][];
    const expected = c['expected'] as number[][];
    const out = kmeansUpdateMeans(data, assignments, k, previousMeans);
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('kmeansFit', () => {
  it('reproduces a hand-rolled assign/update trace from fixed initial means', () => {
    const c = casesFor('kmeansTrace')[0]!;
    const data = c['data'] as number[][];
    const initialMeans = c['initialMeans'] as number[][];
    const steps = c['steps'] as number;
    const expectedMeansHistory = c['expectedMeansHistory'] as number[][][];
    const expectedAssignmentsHistory = c['expectedAssignmentsHistory'] as number[][];
    const expectedDistortionHistory = c['expectedDistortionHistory'] as number[];

    const result = kmeansFit(data, initialMeans, steps);

    expect(result.meansHistory).toHaveLength(expectedMeansHistory.length);
    result.meansHistory.forEach((means, s) =>
      means.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expectedMeansHistory[s]![i]![j]!, 9))),
    );
    result.assignmentsHistory.forEach((a, s) => expect(a).toEqual(expectedAssignmentsHistory[s]));
    result.distortionHistory.forEach((d, s) => expect(d).toBeCloseTo(expectedDistortionHistory[s]!, 9));
  });

  it('has a non-increasing distortion history, the guarantee behind its termination', () => {
    const c = casesFor('kmeansTrace')[0]!;
    const data = c['data'] as number[][];
    const initialMeans = c['initialMeans'] as number[][];
    const result = kmeansFit(data, initialMeans, 10);
    for (let i = 1; i < result.distortionHistory.length; i++) {
      expect(result.distortionHistory[i]!).toBeLessThanOrEqual(result.distortionHistory[i - 1]! + 1e-9);
    }
  });

  it('reports convergence once an update leaves every mean unchanged', () => {
    const c = casesFor('kmeansTrace')[0]!;
    const data = c['data'] as number[][];
    const initialMeans = c['initialMeans'] as number[][];
    const result = kmeansFit(data, initialMeans, 50);
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThan(50);
  });
});

describe('kmeansInit', () => {
  it('returns k distinct rows drawn from the data, deterministically for a given seed', () => {
    const data = [
      [0, 0],
      [1, 0],
      [0, 1],
      [5, 5],
      [6, 5],
      [5, 6],
    ];
    const a = kmeansInit(pcg32(42), data, 3);
    const b = kmeansInit(pcg32(42), data, 3);
    expect(a).toEqual(b);
    expect(a).toHaveLength(3);
    const dataSet = new Set(data.map((row) => JSON.stringify(row)));
    for (const mean of a) expect(dataSet.has(JSON.stringify(mean))).toBe(true);
    const uniqueMeans = new Set(a.map((row) => JSON.stringify(row)));
    expect(uniqueMeans.size).toBe(3);
  });

  it('gives a different seed a real chance of landing on a different initial set', () => {
    const data = Array.from({ length: 20 }, (_, i) => [i, i * 2]);
    const a = kmeansInit(pcg32(1), data, 4);
    const b = kmeansInit(pcg32(2), data, 4);
    expect(JSON.stringify(a)).not.toEqual(JSON.stringify(b));
  });
});
