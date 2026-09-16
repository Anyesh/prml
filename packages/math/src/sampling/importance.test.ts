import { describe, expect, it } from 'vitest';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { normalLogPdf } from '../distributions/normal.js';
import { effectiveSampleSize, importanceEstimate, importanceLogWeights, normalizeImportanceWeights } from './importance.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_importance');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('importanceLogWeights / normalizeImportanceWeights', () => {
  it('matches an independent scipy computation for a fixed sample set', () => {
    const c = casesFor('importanceLogWeights')[0]!;
    const samples = c['samples'] as number[];
    const logWeights = importanceLogWeights(
      samples,
      (z) => normalLogPdf(z, { mu: c['targetMean'] as number, sigma2: (c['targetScale'] as number) ** 2 }),
      (z) => normalLogPdf(z, { mu: c['proposalMean'] as number, sigma2: (c['proposalScale'] as number) ** 2 }),
    );
    (c['expected'] as number[]).forEach((e, i) => expect(closeTo(logWeights[i]!, e)).toBe(true));

    const normCase = casesFor('normalizeImportanceWeights')[0]!;
    const weights = normalizeImportanceWeights(normCase['logWeights'] as number[]);
    (normCase['expected'] as number[]).forEach((e, i) => expect(closeTo(weights[i]!, e)).toBe(true));
    expect(weights.reduce((s, w) => s + w, 0)).toBeCloseTo(1, 9);
  });
});

describe('importanceEstimate', () => {
  it('matches the weighted sum for identity and square test functions', () => {
    for (const c of casesFor('importanceEstimate')) {
      const actual = importanceEstimate(c['values'] as number[], c['weights'] as number[]);
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
  });
});

describe('effectiveSampleSize', () => {
  it('matches 1/sum(w^2) exactly and collapses when one weight dominates', () => {
    const cases = casesFor('effectiveSampleSize');
    expect(cases.length).toBeGreaterThanOrEqual(2);
    for (const c of cases) {
      const actual = effectiveSampleSize(c['weights'] as number[]);
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
    const balanced = effectiveSampleSize(new Array(8).fill(1 / 8));
    expect(balanced).toBeCloseTo(8, 9);
    const degenerate = effectiveSampleSize(casesFor('effectiveSampleSize')[1]!['weights'] as number[]);
    expect(degenerate).toBeLessThan(2);
  });
});
