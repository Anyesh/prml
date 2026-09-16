import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { klDivergence, gaussianKlDivergence } from './divergence.js';
import type { NormalParams } from '../distributions/normal.js';

interface Case {
  readonly fn: string;
  readonly p?: number[] | NormalParams;
  readonly q?: number[] | NormalParams;
  readonly expected: number;
}

const fixture = loadFixture<{ cases: Case[] }>('information');

function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('klDivergence', () => {
  it('matches scipy.stats.entropy(p, q) on every golden case', () => {
    const cases = casesFor('klDivergence');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = klDivergence(c.p as number[], c.q as number[]);
      expect(closeTo(actual, c.expected), `klDivergence(${JSON.stringify(c.p)}, ${JSON.stringify(c.q)})`).toBe(true);
    }
  });

  it('is zero exactly when the two distributions are equal', () => {
    expect(klDivergence([0.5, 0.5], [0.5, 0.5])).toBe(0);
  });
});

describe('gaussianKlDivergence', () => {
  it('matches a numerical integration of p ln(p/q) on every golden case', () => {
    const cases = casesFor('gaussianKlDivergence');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = gaussianKlDivergence(c.p as NormalParams, c.q as NormalParams);
      expect(closeTo(actual, c.expected), `gaussianKlDivergence(${JSON.stringify(c.p)}, ${JSON.stringify(c.q)})`).toBe(true);
    }
  });

  it('is asymmetric', () => {
    const p: NormalParams = { mu: 0, sigma2: 1 };
    const q: NormalParams = { mu: 2, sigma2: 3 };
    expect(gaussianKlDivergence(p, q)).not.toBeCloseTo(gaussianKlDivergence(q, p), 6);
  });
});
