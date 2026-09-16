import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { sequentialMean } from './sequentialMean.js';

interface Case {
  readonly fn: string;
  readonly previous: number;
  readonly n: number;
  readonly x: number;
  readonly expected: number;
}

const fixture = loadFixture<{ cases: Case[] }>('gaussian');

describe('sequentialMean', () => {
  it('reproduces the running mean of a 12-point stream (PRML 2.126), against a directly computed prefix mean', () => {
    for (const c of fixture.cases.filter((c) => c.fn === 'sequentialMean')) {
      expect(sequentialMean(c.previous, c.n, c.x)).toBeCloseTo(c.expected, 9);
    }
  });

  it('equals the batch mean of the whole stream after the last update', () => {
    const stream = [2, 4, 4, 4, 5, 5, 7, 9];
    let running = 0;
    stream.forEach((x, i) => {
      running = sequentialMean(running, i + 1, x);
    });
    const batch = stream.reduce((s, x) => s + x, 0) / stream.length;
    expect(running).toBeCloseTo(batch, 12);
  });
});
