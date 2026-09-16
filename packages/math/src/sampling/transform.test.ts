import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { closeTo, loadFixture } from '../testing/fixture.js';
import {
  boxMullerTrace,
  cauchyPdf,
  cauchyQuantile,
  cauchySample,
  exponentialLogPdf,
  exponentialQuantile,
  exponentialSample,
} from './transform.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_transform');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('exponentialQuantile / exponentialLogPdf', () => {
  it('matches scipy.stats.expon on every golden case', () => {
    const quantileCases = casesFor('exponentialQuantile');
    expect(quantileCases.length).toBeGreaterThan(0);
    for (const c of quantileCases) {
      const actual = exponentialQuantile(c['u'] as number, c['lambda'] as number);
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
    for (const c of casesFor('exponentialLogPdf')) {
      const actual = exponentialLogPdf(c['x'] as number, c['lambda'] as number);
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
  });

  it('exponentialSample only ever consumes one uniform draw and matches the quantile', () => {
    const rng = pcg32(1);
    const u = rng.next();
    const rng2 = pcg32(1);
    expect(exponentialSample(rng2, 2)).toBeCloseTo(exponentialQuantile(u, 2), 12);
  });
});

describe('cauchyQuantile / cauchyPdf', () => {
  it('matches scipy.stats.cauchy on every golden case', () => {
    for (const c of casesFor('cauchyQuantile')) {
      const actual = cauchyQuantile(c['u'] as number, { b: c['b'] as number, c: c['c'] as number });
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
    for (const c of casesFor('cauchyPdf')) {
      const actual = cauchyPdf(c['x'] as number, { b: c['b'] as number, c: c['c'] as number });
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
  });

  it('cauchySample transforms a single uniform draw through cauchyQuantile', () => {
    const params = { b: 1.5, c: -0.5 };
    const rng = pcg32(42);
    const u = rng.next();
    const rng2 = pcg32(42);
    expect(cauchySample(rng2, params)).toBeCloseTo(cauchyQuantile(u, params), 12);
  });
});

describe('boxMullerTrace', () => {
  it('reproduces the reference PCG32 trace bit-for-bit for a fixed seed, including a rejected attempt', () => {
    const cases = casesFor('boxMullerTrace');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const rng = pcg32(c['seed'] as number, c['stream'] as number);
      const trace = boxMullerTrace(rng);
      const expected = c['expected'] as { attempts: { z1: number; z2: number; r2: number; accepted: boolean }[]; y1: number; y2: number };
      expect(trace.attempts).toHaveLength(expected.attempts.length);
      trace.attempts.forEach((a, i) => {
        expect(closeTo(a.z1, expected.attempts[i]!.z1)).toBe(true);
        expect(closeTo(a.z2, expected.attempts[i]!.z2)).toBe(true);
        expect(a.accepted).toBe(expected.attempts[i]!.accepted);
      });
      expect(closeTo(trace.y1, expected.y1)).toBe(true);
      expect(closeTo(trace.y2, expected.y2)).toBe(true);
    }
    // At least one fixture seed must exercise a rejection, or the geometry Figure 11.3
    // needs (a point outside the unit disk) would never be tested.
    expect(cases.some((c) => (c['expected'] as { attempts: unknown[] }).attempts.length > 1)).toBe(true);
  });

  it('y1 and y2 land near the standard normal mean and variance over many draws', () => {
    const rng = pcg32(777);
    const n = 20000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const { y1, y2 } = boxMullerTrace(rng);
      sum += y1 + y2;
      sumSq += y1 * y1 + y2 * y2;
    }
    const count = 2 * n;
    const empMean = sum / count;
    const empVar = sumSq / count - empMean * empMean;
    expect(Math.abs(empMean)).toBeLessThan(0.05);
    expect(Math.abs(empVar - 1)).toBeLessThan(0.05);
  });
});
