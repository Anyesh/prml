import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import type { Mat, Vec } from '../types.js';
import { principalCurveFit, projectToPolyline, swissRollPoint } from './manifold.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('manifold');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('swissRollPoint', () => {
  it('matches the standard swiss-roll parametrisation for every (t, height) pair', () => {
    for (const c of casesFor('swissRollPoint')) {
      const result = swissRollPoint(c['t'] as number, c['height'] as number);
      const expected = c['expected'] as number[];
      result.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
    }
  });
});

describe('projectToPolyline', () => {
  it('finds the nearest point on a polyline and its arclength parameter', () => {
    const c = casesFor('projectToPolyline')[0]!;
    const result = projectToPolyline(c['point'] as Vec, c['polyline'] as Mat);
    const expected = c['expected'] as { arclength: number; distance: number };
    expect(result.arclength).toBeCloseTo(expected.arclength, 9);
    expect(result.distance).toBeCloseTo(expected.distance, 9);
  });
});

describe('principalCurveFit', () => {
  it('reproduces a hand-rolled self-consistency iterate trace (PRML 12.92)', () => {
    const c = casesFor('principalCurveFit')[0]!;
    const data = c['data'] as Mat;
    const bandwidth = c['bandwidth'] as number;
    let curve = c['initialCurve'] as Mat;
    const expectedTrace = c['expectedTrace'] as number[][][];

    for (const expectedCurve of expectedTrace) {
      const result = principalCurveFit(data, curve, 1, bandwidth);
      const nextCurve = result.curveHistory[result.curveHistory.length - 1]!;
      nextCurve.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expectedCurve[i]![j]!, 9)));
      curve = nextCurve;
    }
  });
});
