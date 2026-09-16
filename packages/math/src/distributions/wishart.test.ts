import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { loadFixture } from '../testing/fixture.js';
import {
  wishartExpectedLogDet,
  wishartLogPdf,
  wishartMean,
  wishartPdf,
  wishartSample,
} from './wishart.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('wishart');
function findCase(fn: string): Case {
  const c = fixture.cases.find((c) => c.fn === fn);
  if (!c) throw new Error(`no fixture case for fn "${fn}"`);
  return c;
}

describe('wishartLogPdf / wishartPdf', () => {
  it('matches scipy.stats.wishart at a non-integer nu', () => {
    const logC = findCase('wishartLogPdf');
    const params = logC['params'] as { scale: number[][]; nu: number };
    const x = logC['x'] as number[][];
    expect(wishartLogPdf(x, params)).toBeCloseTo(logC['expected'] as number, 9);

    const pdfC = findCase('wishartPdf');
    expect(wishartPdf(x, params)).toBeCloseTo(pdfC['expected'] as number, 9);
  });
});

describe('wishartMean', () => {
  it('is nu * scale, matching scipy', () => {
    const c = findCase('wishartMean');
    const params = c['params'] as { scale: number[][]; nu: number };
    const out = wishartMean(params);
    const expected = c['expected'] as number[][];
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('wishartExpectedLogDet', () => {
  it('matches PRML 10.65 computed independently via scipy digamma', () => {
    const c = findCase('wishartExpectedLogDet');
    const params = c['params'] as { scale: number[][]; nu: number };
    expect(wishartExpectedLogDet(params)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('wishartSample', () => {
  it('the Bartlett decomposition matches the analytic mean nu*scale within a standard-error-derived tolerance', () => {
    // No fixture: the README's sampler policy checks a large sample's mean against the
    // distribution's own analytic mean, using these input parameters directly.
    const scale = [
      [2.0, 0.3, 0.1],
      [0.3, 1.5, 0.2],
      [0.1, 0.2, 1.0],
    ];
    const nu = 6.5;
    const d = scale.length;
    const n = 50_000;
    const rng = pcg32(555, 3);

    const empMean: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let s = 0; s < n; s++) {
      const sample = wishartSample(rng, { scale, nu });
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          empMean[i]![j]! += sample[i]![j]! / n;
        }
      }
    }

    // Var(Lambda_ij) for a Wishart(scale, nu) is nu*(scale_ij^2 + scale_ii*scale_jj),
    // the standard Wishart second-moment formula; SE of the sample mean over n draws
    // is that divided by n, then square-rooted.
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        const trueMean = nu * scale[i]![j]!;
        const variance = nu * (scale[i]![j]! ** 2 + scale[i]![i]! * scale[j]![j]!);
        const se = Math.sqrt(variance / n);
        expect(Math.abs(empMean[i]![j]! - trueMean)).toBeLessThan(6 * se);
      }
    }
  });
});
