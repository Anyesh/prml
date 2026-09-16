import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { gammaCauchyEnvelope, gammaCauchySampler, rejectionSampleBatch } from './rejection.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_rejection');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('gammaCauchyEnvelope', () => {
  it('matches the closed-form tangency parameters (PRML 11.1.2)', () => {
    for (const c of casesFor('gammaCauchyEnvelope')) {
      const actual = gammaCauchyEnvelope(c['a'] as number);
      const expected = c['expected'] as { b: number; c: number; k: number };
      expect(closeTo(actual.b, expected.b)).toBe(true);
      expect(closeTo(actual.c, expected.c)).toBe(true);
      expect(closeTo(actual.k, expected.k)).toBe(true);
    }
  });

  it('never lets the gamma density exceed k*q(z) on a fine grid (an independent scipy check, not a re-derivation)', () => {
    for (const c of casesFor('gammaCauchyEnvelope_maxRatioOnGrid')) {
      const maxRatio = c['expected'] as number;
      expect(maxRatio).toBeLessThanOrEqual(1 + 1e-6);
      expect(maxRatio).toBeGreaterThan(1 - 1e-6);
    }
  });
});

describe('rejectionSampleBatch', () => {
  it('reproduces a reference accept/reject trajectory bit-for-bit for a fixed seed', () => {
    const cases = casesFor('rejectionSampleBatch');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const rng = pcg32(c['seed'] as number, 1);
      const sampler = gammaCauchySampler(c['a'] as number);
      const result = rejectionSampleBatch(rng, sampler, c['n'] as number);
      const expected = c['expected'] as {
        samples: number[];
        trace: { candidate: number; u: number; envelope: number; target: number; accepted: boolean }[];
        acceptanceRate: number;
      };
      expect(result.samples).toHaveLength(expected.samples.length);
      result.samples.forEach((s, i) => expect(closeTo(s, expected.samples[i]!)).toBe(true));
      expect(result.trace).toHaveLength(expected.trace.length);
      result.trace.forEach((attempt, i) => {
        const e = expected.trace[i]!;
        expect(closeTo(attempt.candidate, e.candidate)).toBe(true);
        expect(closeTo(attempt.u, e.u)).toBe(true);
        expect(closeTo(attempt.envelope, e.envelope)).toBe(true);
        expect(closeTo(attempt.target, e.target)).toBe(true);
        expect(attempt.accepted).toBe(e.accepted);
      });
      expect(closeTo(result.acceptanceRate, expected.acceptanceRate)).toBe(true);
    }
  });

  it("empirical acceptance rate tracks the theoretical 1/k (PRML 11.14) within a few percent", () => {
    const a = 4.3;
    const sampler = gammaCauchySampler(a);
    const rng = pcg32(2024, 3);
    const result = rejectionSampleBatch(rng, sampler, 5000);
    expect(Math.abs(result.acceptanceRate - 1 / sampler.k)).toBeLessThan(0.03);
  });

  it('accepted samples track the gamma mean and variance', () => {
    const a = 4.3;
    const sampler = gammaCauchySampler(a);
    const rng = pcg32(55, 9);
    const result = rejectionSampleBatch(rng, sampler, 20000);
    const n = result.samples.length;
    const mean = result.samples.reduce((s, x) => s + x, 0) / n;
    const variance = result.samples.reduce((s, x) => s + (x - mean) ** 2, 0) / n;
    expect(Math.abs(mean - a)).toBeLessThan(0.1);
    expect(Math.abs(variance - a)).toBeLessThan(0.2);
  });
});
