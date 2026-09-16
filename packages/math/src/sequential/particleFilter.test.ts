import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  particleFilterPredict,
  particleFilterWeights,
  systematicResample,
  systematicResampleIndices,
} from './particleFilter.js';
import { pcg32 } from '../rng.js';
import type { Vec } from '../types.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('particle_filter');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('particleFilterWeights', () => {
  it('matches raw-likelihood normalisation (PRML 13.118)', () => {
    for (const c of casesFor('particleFilterWeights')) {
      const out = particleFilterWeights(c['raw'] as Vec);
      const expected = c['expected'] as Vec;
      out.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
    }
  });

  it('always sums to one', () => {
    for (const c of casesFor('particleFilterWeights')) {
      const out = particleFilterWeights(c['raw'] as Vec);
      expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    }
  });
});

describe('systematicResampleIndices', () => {
  it('matches the plain-numpy systematic resampling scheme from a fixed uniform draw', () => {
    for (const c of casesFor('systematicResampleIndices')) {
      const out = systematicResampleIndices(c['u0'] as number, c['weights'] as Vec, c['L'] as number);
      expect(out).toEqual(c['expected'] as number[]);
    }
  });
});

describe('systematicResample', () => {
  it('is deterministic for a fixed seed and draws exactly L indices, all valid', () => {
    const weights: Vec = [0.1, 0.6, 0.3];
    const a = systematicResample(pcg32(20260917), weights, 50);
    const b = systematicResample(pcg32(20260917), weights, 50);
    expect(a).toEqual(b);
    expect(a).toHaveLength(50);
    for (const idx of a) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(weights.length);
    }
  });

  it('over many draws, resampling frequency tracks the input weights', () => {
    const weights: Vec = [0.1, 0.7, 0.2];
    const indices = systematicResample(pcg32(7), weights, 10000);
    const counts = [0, 0, 0];
    for (const idx of indices) counts[idx]! += 1;
    counts.forEach((count, k) => {
      expect(count / indices.length).toBeCloseTo(weights[k]!, 1);
    });
  });

  it('a particle with zero weight is never resampled', () => {
    const weights: Vec = [0, 0.5, 0.5];
    const indices = systematicResample(pcg32(3), weights, 200);
    expect(indices.every((idx) => idx !== 0)).toBe(true);
  });
});

describe('particleFilterPredict', () => {
  it('draws L new particles, each from the transition kernel of a resampled ancestor', () => {
    const particles = [[0], [10], [20]];
    const weights: Vec = [1, 0, 0];
    const rng = pcg32(11);
    const predicted = particleFilterPredict(rng, particles, weights, (_rng, z) => [z[0]! + 1]);
    // Every ancestor is particle 0 (the only nonzero weight), so every predicted particle
    // must be exactly one step ahead of it.
    expect(predicted.every((p) => p[0] === 1)).toBe(true);
    expect(predicted).toHaveLength(particles.length);
  });
});
