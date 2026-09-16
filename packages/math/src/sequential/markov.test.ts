import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { markovChainLogLikelihood, markovChainSample } from './markov.js';
import { pcg32 } from '../rng.js';
import type { Mat, Vec } from '../types.js';

interface Case {
  readonly fn: string;
  readonly pi: Vec;
  readonly A: Mat;
  readonly states: readonly number[];
  readonly expected: number;
}

const fixture = loadFixture<{ cases: Case[] }>('markov');

describe('markovChainLogLikelihood', () => {
  it('matches a plain-numpy sum of log transition probabilities (PRML 13.2)', () => {
    for (const c of fixture.cases) {
      expect(markovChainLogLikelihood(c.pi, c.A, c.states)).toBeCloseTo(c.expected, 9);
    }
  });

  it('returns -Infinity for a path that crosses a structurally zero transition', () => {
    const pi: Vec = [1, 0];
    const A: Mat = [
      [1, 0],
      [0, 1],
    ];
    expect(markovChainLogLikelihood(pi, A, [0, 1])).toBe(-Infinity);
  });
});

describe('markovChainSample', () => {
  it('always starts in a state with nonzero pi and only ever takes nonzero transitions', () => {
    const pi: Vec = [0, 0.4, 0.6];
    const A: Mat = [
      [1, 0, 0],
      [0, 0.5, 0.5],
      [0.3, 0, 0.7],
    ];
    const rng = pcg32(20260917);
    const path = markovChainSample(rng, pi, A, 200);
    expect(path[0]).not.toBe(0);
    for (let t = 1; t < path.length; t++) {
      expect(A[path[t - 1]!]![path[t]!]).toBeGreaterThan(0);
    }
  });

  it('is deterministic for a fixed seed', () => {
    const pi: Vec = [0.5, 0.5];
    const A: Mat = [
      [0.7, 0.3],
      [0.4, 0.6],
    ];
    const a = markovChainSample(pcg32(7), pi, A, 20);
    const b = markovChainSample(pcg32(7), pi, A, 20);
    expect(a).toEqual(b);
  });
});
