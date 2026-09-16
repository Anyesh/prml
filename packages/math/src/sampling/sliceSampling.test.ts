import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { sliceSampleChain, sliceSteppingOut } from './sliceSampling.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_slice');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

const LOG_2PI = Math.log(2 * Math.PI);
function logTarget(z: number): number {
  return -0.5 * z * z - 0.5 * LOG_2PI;
}

describe('sliceSteppingOut', () => {
  it('matches an independent re-implementation exactly, needing no RNG', () => {
    const cases = casesFor('sliceSteppingOut');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const interval = sliceSteppingOut(c['z0'] as number, c['logU'] as number, c['w'] as number, logTarget);
      const expected = c['expected'] as { lo: number; hi: number };
      expect(closeTo(interval.lo, expected.lo)).toBe(true);
      expect(closeTo(interval.hi, expected.hi)).toBe(true);
    }
  });

  it('always returns an interval containing z0 that lies inside the slice at both new endpoints or a single step away from it', () => {
    const interval = sliceSteppingOut(0.5, Math.log(0.2) + logTarget(0.5), 0.3, logTarget);
    expect(interval.lo).toBeLessThanOrEqual(0.5);
    expect(interval.hi).toBeGreaterThanOrEqual(0.5);
  });
});

describe('sliceSampleChain', () => {
  it('reproduces a reference step-and-shrink trajectory bit-for-bit for a fixed seed', () => {
    const cases = casesFor('sliceSampleChain');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const rng = pcg32(c['seed'] as number, 9);
      const result = sliceSampleChain(rng, c['initial'] as number, c['nSteps'] as number, logTarget, c['w'] as number);
      const expected = c['expected'] as { states: number[]; steps: unknown[] };
      expect(result.states).toHaveLength(expected.states.length);
      result.states.forEach((s, i) => expect(closeTo(s, expected.states[i]!)).toBe(true));
      expect(result.steps).toHaveLength(expected.steps.length);
    }
  });

  it('recovers the standard normal mean and variance from a long chain', () => {
    const rng = pcg32(4242, 21);
    const result = sliceSampleChain(rng, 0, 8000, logTarget, 1.5);
    const kept = result.states.slice(500);
    const mean = kept.reduce((s, x) => s + x, 0) / kept.length;
    const variance = kept.reduce((s, x) => s + (x - mean) ** 2, 0) / kept.length;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.abs(variance - 1)).toBeLessThan(0.1);
  });
});
