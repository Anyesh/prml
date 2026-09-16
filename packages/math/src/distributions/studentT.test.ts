import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import {
  studentTLogPdf,
  studentTPdf,
  studentTCdf,
  studentTSample,
  multivariateTLogPdf,
  type StudentTParams,
  type MultivariateTParams,
} from './studentT.js';

interface UnivariateCase {
  readonly fn: string;
  readonly x: number;
  readonly params: StudentTParams;
  readonly expected: number;
}

interface MultivariateCase {
  readonly fn: 'multivariateLogPdf';
  readonly x: number[];
  readonly params: MultivariateTParams;
  readonly expected: number;
}

const fixture = loadFixture<{ cases: (UnivariateCase | MultivariateCase)[] }>('studentT');

function casesFor<T>(fn: string): T[] {
  return fixture.cases.filter((c) => c.fn === fn) as T[];
}

describe('studentTLogPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor<UnivariateCase>('logPdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = studentTLogPdf(c.x, c.params);
      expect(closeTo(actual, c.expected), `logPdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('studentTPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor<UnivariateCase>('pdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = studentTPdf(c.x, c.params);
      expect(closeTo(actual, c.expected), `pdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('studentTCdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor<UnivariateCase>('cdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = studentTCdf(c.x, c.params);
      expect(closeTo(actual, c.expected), `cdf(${c.x}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('multivariateTLogPdf', () => {
  it('matches a numpy/scipy reference on every golden case', () => {
    // Depends on A3's `linalg` (solve, logDet); expected to throw NotImplemented until that lands.
    const cases = casesFor<MultivariateCase>('multivariateLogPdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = multivariateTLogPdf(c.x, c.params);
      expect(closeTo(actual, c.expected), `multivariateLogPdf(${JSON.stringify(c.x)}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('studentTSample', () => {
  it('draws values whose sample mean tracks mu', () => {
    // Depends on A3's `standardNormal` via gammaSample; expected to throw until that lands.
    const rng = pcg32(2024);
    const p: StudentTParams = { mu: 2, scale2: 1, nu: 10 };
    const n = 20000;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += studentTSample(rng, p);
    }
    expect(Math.abs(sum / n - p.mu)).toBeLessThan(0.15);
  });
});
