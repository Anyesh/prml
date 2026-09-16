import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { gammaLogPdf } from '../distributions/gamma.js';
import { exponentialLogPdf, exponentialSample } from './transform.js';
import { samplingImportanceResampling } from './sir.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_sir');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('samplingImportanceResampling', () => {
  it('reproduces a reference two-stage trajectory bit-for-bit for a fixed seed', () => {
    const cases = casesFor('samplingImportanceResampling');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const rng = pcg32(c['seed'] as number, 2);
      const a = c['a'] as number;
      const result = samplingImportanceResampling(rng, {
        sampleProposal: (r) => exponentialSample(r, 1),
        proposalLogPdf: (z) => exponentialLogPdf(z, 1),
        targetLogPdf: (z) => gammaLogPdf(z, { shape: a, rate: 1 }),
        l: c['l'] as number,
      });
      const expected = c['expected'] as { proposalSamples: number[]; weights: number[]; resampled: number[] };
      result.proposalSamples.forEach((z, i) => expect(closeTo(z, expected.proposalSamples[i]!)).toBe(true));
      result.weights.forEach((w, i) => expect(closeTo(w, expected.weights[i]!)).toBe(true));
      result.resampled.forEach((z, i) => expect(closeTo(z, expected.resampled[i]!)).toBe(true));
    }
  });

  it('as L grows, the resampled mean approaches the gamma target mean (PRML 11.26)', () => {
    const a = 4.0;
    const rng = pcg32(303, 8);
    const result = samplingImportanceResampling(rng, {
      sampleProposal: (r) => exponentialSample(r, 1 / a),
      proposalLogPdf: (z) => exponentialLogPdf(z, 1 / a),
      targetLogPdf: (z) => gammaLogPdf(z, { shape: a, rate: 1 }),
      l: 20000,
    });
    const mean = result.resampled.reduce((s, x) => s + x, 0) / result.resampled.length;
    expect(Math.abs(mean - a)).toBeLessThan(0.3);
  });
});
