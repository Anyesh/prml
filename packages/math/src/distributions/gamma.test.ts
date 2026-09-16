import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import { gammaLogPdf, gammaPdf, gammaCdf, gammaSample, gammaMean, gammaVariance, type GammaParams } from './gamma.js';

interface Case {
  readonly fn: string;
  readonly x?: number;
  readonly params: GammaParams;
  readonly expected: number;
}

const fixture = loadFixture<{ cases: Case[] }>('gamma');

function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('gammaLogPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('logPdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = gammaLogPdf(c.x!, c.params);
      expect(closeTo(actual, c.expected), `logPdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('gammaPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('pdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = gammaPdf(c.x!, c.params);
      expect(closeTo(actual, c.expected), `pdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('gammaCdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('cdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = gammaCdf(c.x!, c.params);
      expect(closeTo(actual, c.expected), `cdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('gammaMean', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('mean');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = gammaMean(c.params);
      expect(closeTo(actual, c.expected), `mean(${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('gammaVariance', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('variance');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = gammaVariance(c.params);
      expect(closeTo(actual, c.expected), `variance(${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('gammaSample', () => {
  it('draws positive values whose sample mean tracks gammaMean', () => {
    // Depends on A3's `standardNormal` in rng.ts; expected to throw NotImplemented until that lands.
    const rng = pcg32(99);
    const p: GammaParams = { shape: 3, rate: 2 };
    const n = 20000;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const s = gammaSample(rng, p);
      expect(s).toBeGreaterThan(0);
      sum += s;
    }
    expect(Math.abs(sum / n - gammaMean(p))).toBeLessThan(0.05);
  });

  it('boosts shape < 1 without looping forever', () => {
    const rng = pcg32(55);
    const p: GammaParams = { shape: 0.3, rate: 1 };
    for (let i = 0; i < 100; i++) {
      expect(gammaSample(rng, p)).toBeGreaterThan(0);
    }
  });
});
