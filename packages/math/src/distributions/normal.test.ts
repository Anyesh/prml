import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import { normalLogPdf, normalPdf, normalCdf, normalQuantile, normalSample, type NormalParams } from './normal.js';

interface Case {
  readonly fn: string;
  readonly x?: number;
  readonly q?: number;
  readonly params: NormalParams;
  readonly expected: number;
}

const fixture = loadFixture<{ cases: Case[] }>('normal');

function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('normalLogPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('logPdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = normalLogPdf(c.x!, c.params);
      expect(closeTo(actual, c.expected), `logPdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('normalPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('pdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = normalPdf(c.x!, c.params);
      expect(closeTo(actual, c.expected), `pdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('normalCdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('cdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = normalCdf(c.x!, c.params);
      expect(closeTo(actual, c.expected), `cdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('normalQuantile', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('quantile');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = normalQuantile(c.q!, c.params);
      expect(closeTo(actual, c.expected), `quantile(${c.q}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('normalSample', () => {
  it('draws from the correct mean and variance', () => {
    // Depends on A3's `standardNormal` in rng.ts; expected to throw NotImplemented until that lands.
    const rng = pcg32(12345);
    const p: NormalParams = { mu: 3, sigma2: 4 };
    const n = 20000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const s = normalSample(rng, p);
      sum += s;
      sumSq += s * s;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    expect(Math.abs(mean - p.mu)).toBeLessThan(0.1);
    expect(Math.abs(variance - p.sigma2)).toBeLessThan(0.3);
  });
});
