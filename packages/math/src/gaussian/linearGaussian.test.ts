import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { linearGaussianMarginal, linearGaussianPosterior } from './linearGaussian.js';

interface MvnExpected {
  readonly mean: readonly number[];
  readonly cov: readonly (readonly number[])[];
}

interface Case {
  readonly fn: string;
  readonly prior?: { mean: number[]; precision: number[][] };
  readonly likelihood?: { a: number[][]; b: number[]; precision: number[][] };
  readonly y?: number[];
  readonly expected: MvnExpected;
}

const fixture = loadFixture<{ cases: Case[] }>('gaussian');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

function expectClose(actual: MvnExpected, expected: MvnExpected) {
  actual.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, 9));
  actual.cov.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected.cov[i]![j]!, 9)));
}

describe('linearGaussianMarginal', () => {
  it('matches the direct numpy computation of p(y) (PRML 2.114-2.115), square and rectangular A', () => {
    for (const c of casesFor('linearGaussianMarginal')) {
      const out = linearGaussianMarginal(c.prior!, c.likelihood!);
      expectClose(out, c.expected);
    }
  });

  it('agrees with the joint-precision route (2.104-2.105) that never calls the formula under test', () => {
    for (const c of casesFor('linearGaussianMarginal_jointCheck')) {
      const out = linearGaussianMarginal(c.prior!, c.likelihood!);
      expectClose(out, c.expected);
    }
  });
});

describe('linearGaussianPosterior', () => {
  it('matches the direct numpy computation of p(x|y) (PRML 2.116-2.117), square and rectangular A', () => {
    for (const c of casesFor('linearGaussianPosterior')) {
      const out = linearGaussianPosterior(c.prior!, c.likelihood!, c.y!);
      expectClose(out, c.expected);
    }
  });
});
