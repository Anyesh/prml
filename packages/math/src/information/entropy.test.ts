import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { discreteEntropy, differentialEntropyGaussian } from './entropy.js';

interface Case {
  readonly fn: string;
  readonly p?: number[];
  readonly sigma2?: number;
  readonly expected: number;
}

const fixture = loadFixture<{ cases: Case[] }>('information');

function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('discreteEntropy', () => {
  it('matches scipy.stats.entropy on every golden case', () => {
    const cases = casesFor('discreteEntropy');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = discreteEntropy(c.p!);
      expect(closeTo(actual, c.expected), `discreteEntropy(${JSON.stringify(c.p)})`).toBe(true);
    }
  });

  it('is zero for a one-hot distribution', () => {
    expect(discreteEntropy([1, 0, 0])).toBe(0);
  });
});

describe('differentialEntropyGaussian', () => {
  it('matches scipy.stats.norm(...).entropy() on every golden case', () => {
    const cases = casesFor('differentialEntropyGaussian');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = differentialEntropyGaussian(c.sigma2!);
      expect(closeTo(actual, c.expected), `differentialEntropyGaussian(${c.sigma2})`).toBe(true);
    }
  });

  it('goes negative below variance 1/(2 pi e), unlike discrete entropy', () => {
    expect(differentialEntropyGaussian(1 / (2 * Math.PI * Math.E) - 0.01)).toBeLessThan(0);
  });
});
