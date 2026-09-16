import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import {
  bernoulliLogPmf,
  bernoulliSample,
  binomialLogPmf,
  binomialPmf,
  multinomialLogPmf,
  multinomialSample,
  logBinomialCoefficient,
  type BernoulliParams,
  type BinomialParams,
  type MultinomialParams,
} from './discrete.js';

interface BernoulliCase {
  readonly fn: 'bernoulliLogPmf';
  readonly x: 0 | 1;
  readonly params: BernoulliParams;
  readonly expected: number;
}

interface BinomialCase {
  readonly fn: 'binomialLogPmf' | 'binomialPmf';
  readonly m: number;
  readonly params: BinomialParams;
  readonly expected: number;
}

interface MultinomialCase {
  readonly fn: 'multinomialLogPmf';
  readonly counts: number[];
  readonly params: MultinomialParams;
  readonly expected: number;
}

interface CoefficientCase {
  readonly fn: 'logBinomialCoefficient';
  readonly n: number;
  readonly k: number;
  readonly expected: number;
}

type Case = BernoulliCase | BinomialCase | MultinomialCase | CoefficientCase;

const fixture = loadFixture<{ cases: Case[] }>('discrete');

function casesFor<T>(fn: string): T[] {
  return fixture.cases.filter((c) => c.fn === fn) as T[];
}

describe('bernoulliLogPmf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor<BernoulliCase>('bernoulliLogPmf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = bernoulliLogPmf(c.x, c.params);
      expect(closeTo(actual, c.expected), `bernoulliLogPmf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });

  it('returns -Infinity for an impossible outcome at the boundary', () => {
    expect(bernoulliLogPmf(1, { mu: 0 })).toBe(-Infinity);
    expect(bernoulliLogPmf(0, { mu: 1 })).toBe(-Infinity);
  });
});

describe('binomialLogPmf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor<BinomialCase>('binomialLogPmf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = binomialLogPmf(c.m, c.params);
      expect(closeTo(actual, c.expected), `binomialLogPmf(${c.m}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('binomialPmf', () => {
  it('matches scipy on every golden case, including n = 1000 where a naive factorial overflows', () => {
    const cases = casesFor<BinomialCase>('binomialPmf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = binomialPmf(c.m, c.params);
      expect(closeTo(actual, c.expected), `binomialPmf(${c.m}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('multinomialLogPmf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor<MultinomialCase>('multinomialLogPmf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = multinomialLogPmf(c.counts, c.params);
      expect(closeTo(actual, c.expected), `multinomialLogPmf(${JSON.stringify(c.counts)}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('logBinomialCoefficient', () => {
  it('matches a gammaln-based scipy reference, including n = 1000 and n = 170', () => {
    const cases = casesFor<CoefficientCase>('logBinomialCoefficient');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = logBinomialCoefficient(c.n, c.k);
      expect(closeTo(actual, c.expected), `logBinomialCoefficient(${c.n}, ${c.k})`).toBe(true);
    }
  });
});

describe('bernoulliSample', () => {
  it('draws 1 with frequency mu', () => {
    const rng = pcg32(101);
    const p: BernoulliParams = { mu: 0.3 };
    const n = 20000;
    let ones = 0;
    for (let i = 0; i < n; i++) {
      const x = bernoulliSample(rng, p);
      expect(x === 0 || x === 1).toBe(true);
      ones += x;
    }
    expect(Math.abs(ones / n - p.mu)).toBeLessThan(0.02);
  });
});

describe('multinomialSample', () => {
  it('draws counts that sum to n and track the given probs', () => {
    const rng = pcg32(202);
    const p: MultinomialParams = { n: 20, probs: [0.2, 0.3, 0.5] };
    const trials = 5000;
    const totals = [0, 0, 0];
    for (let i = 0; i < trials; i++) {
      const counts = multinomialSample(rng, p);
      expect(counts.length).toBe(3);
      const sum = counts.reduce((a, b) => a + b, 0);
      expect(sum).toBe(p.n);
      for (let k = 0; k < 3; k++) totals[k]! += counts[k]!;
    }
    for (let k = 0; k < 3; k++) {
      const frac = totals[k]! / (trials * p.n);
      expect(Math.abs(frac - p.probs[k]!)).toBeLessThan(0.02);
    }
  });
});
