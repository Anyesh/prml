import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import {
  dirichletLogPdf,
  dirichletPdf,
  dirichletSample,
  dirichletMean,
  dirichletExpectedLog,
  type DirichletParams,
} from './dirichlet.js';

interface Case {
  readonly fn: string;
  readonly x?: number[];
  readonly params: DirichletParams;
  readonly expected: number | number[];
}

const fixture = loadFixture<{ cases: Case[] }>('dirichlet');

function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('dirichletLogPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('logPdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = dirichletLogPdf(c.x!, c.params);
      expect(closeTo(actual, c.expected as number), `logPdf(${JSON.stringify(c.x)}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('dirichletPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('pdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = dirichletPdf(c.x!, c.params);
      expect(closeTo(actual, c.expected as number), `pdf(${JSON.stringify(c.x)}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('dirichletMean', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor('mean');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = dirichletMean(c.params);
      const expected = c.expected as number[];
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < expected.length; i++) {
        expect(closeTo(actual[i]!, expected[i]!), `mean(${JSON.stringify(c.params)})[${i}]`).toBe(true);
      }
    }
  });
});

describe('dirichletExpectedLog', () => {
  it('matches psi(alpha_k) - psi(sum alpha) computed by scipy', () => {
    const cases = casesFor('expectedLog');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = dirichletExpectedLog(c.params);
      const expected = c.expected as number[];
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < expected.length; i++) {
        expect(closeTo(actual[i]!, expected[i]!), `expectedLog(${JSON.stringify(c.params)})[${i}]`).toBe(true);
      }
    }
  });
});

describe('dirichletSample', () => {
  it('draws points on the simplex whose average tracks dirichletMean', () => {
    const rng = pcg32(4242);
    const p: DirichletParams = { alpha: [2, 3, 5] };
    const n = 20000;
    const sums = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const s = dirichletSample(rng, p);
      expect(s.length).toBe(3);
      let total = 0;
      for (const v of s) {
        expect(v).toBeGreaterThanOrEqual(0);
        total += v;
      }
      expect(closeTo(total, 1, 1e-9)).toBe(true);
      for (let k = 0; k < 3; k++) sums[k]! += s[k]!;
    }
    const mean = dirichletMean(p);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(sums[k]! / n - mean[k]!)).toBeLessThan(0.02);
    }
  });
});
