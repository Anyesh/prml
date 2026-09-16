import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from './testing/fixture.js';
import {
  logGamma,
  digamma,
  logBeta,
  logMultivariateGamma,
  erf,
  erfc,
  gammaincLower,
  betainc,
  besselI,
} from './special.js';

interface SpecialCase {
  readonly fn: string;
  readonly args: readonly number[];
  readonly expected: number;
}

const fixture = loadFixture<{ cases: SpecialCase[] }>('special');

const impls: Record<string, (...args: number[]) => number> = {
  logGamma,
  digamma,
  logBeta,
  logMultivariateGamma,
  erf,
  erfc,
  gammaincLower,
  betainc,
  besselI,
};

function casesFor(fn: string): SpecialCase[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe.each(Object.keys(impls))('%s', (fn) => {
  const impl = impls[fn]!;
  it('matches scipy on every golden case', () => {
    const cases = casesFor(fn);
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = impl(...c.args);
      expect(
        closeTo(actual, c.expected),
        `${fn}(${c.args.join(', ')}) = ${actual}, expected ${c.expected}`,
      ).toBe(true);
    }
  });
});
