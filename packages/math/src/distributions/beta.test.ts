import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import {
  betaLogPdf,
  betaPdf,
  betaCdf,
  betaSample,
  betaMean,
  betaVariance,
  betaPosterior,
  type BetaParams,
} from './beta.js';

interface Case {
  readonly fn: string;
  readonly x?: number;
  readonly params?: BetaParams;
  readonly prior?: BetaParams;
  readonly successes?: number;
  readonly failures?: number;
  readonly expected: number | { a: number; b: number };
}

const fixture = loadFixture<{ cases: Case[] }>('beta');

function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('betaLogPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('logPdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = betaLogPdf(c.x!, c.params!);
      expect(closeTo(actual, c.expected as number), `logPdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('betaPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('pdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = betaPdf(c.x!, c.params!);
      expect(closeTo(actual, c.expected as number), `pdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('betaCdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('cdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = betaCdf(c.x!, c.params!);
      expect(closeTo(actual, c.expected as number), `cdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('betaMean', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('mean');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = betaMean(c.params!);
      expect(closeTo(actual, c.expected as number), `mean(${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('betaVariance', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('variance');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = betaVariance(c.params!);
      expect(closeTo(actual, c.expected as number), `variance(${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('betaPosterior', () => {
  it('is pure addition of counts onto the prior, matching the golden cases', () => {
    const cases = casesFor('posterior');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = betaPosterior(c.prior!, c.successes!, c.failures!);
      const expected = c.expected as { a: number; b: number };
      expect(actual.a).toBe(expected.a);
      expect(actual.b).toBe(expected.b);
    }
  });

  it('does not mutate the prior', () => {
    const prior: BetaParams = { a: 2, b: 5 };
    betaPosterior(prior, 3, 1);
    expect(prior).toEqual({ a: 2, b: 5 });
  });
});

describe('betaSample', () => {
  it('draws values whose sample mean tracks betaMean', () => {
    // Depends on A3's `standardNormal`/`pcg32` via gammaSample; expected to throw until that lands.
    const rng = pcg32(777);
    const p: BetaParams = { a: 2, b: 5 };
    const n = 20000;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const s = betaSample(rng, p);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
      sum += s;
    }
    expect(Math.abs(sum / n - betaMean(p))).toBeLessThan(0.02);
  });
});
