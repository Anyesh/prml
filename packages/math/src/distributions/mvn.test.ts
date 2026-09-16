import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { loadFixture } from '../testing/fixture.js';
import {
  mvnConditional,
  mvnCovarianceEllipse,
  mvnLogPdf,
  mvnMarginal,
  mvnPdf,
  mvnSample,
} from './mvn.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('mvn');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('mvnLogPdf / mvnPdf', () => {
  it('matches scipy.stats.multivariate_normal at several points', () => {
    for (const c of casesFor('mvnLogPdf')) {
      const params = c['params'] as { mean: number[]; cov: number[][] };
      const x = c['x'] as number[];
      expect(mvnLogPdf(x, params)).toBeCloseTo(c['expected'] as number, 9);
    }
    for (const c of casesFor('mvnPdf')) {
      const params = c['params'] as { mean: number[]; cov: number[][] };
      const x = c['x'] as number[];
      expect(mvnPdf(x, params)).toBeCloseTo(c['expected'] as number, 9);
    }
  });

  it('pdf is exp(logpdf)', () => {
    const c = casesFor('mvnLogPdf')[0]!;
    const params = c['params'] as { mean: number[]; cov: number[][] };
    const x = c['x'] as number[];
    expect(mvnPdf(x, params)).toBeCloseTo(Math.exp(mvnLogPdf(x, params)), 9);
  });
});

describe('mvnMarginal', () => {
  it('follows the order of `keep`, not ascending index order (PRML 2.98)', () => {
    const c = casesFor('mvnMarginal')[0]!;
    const params = c['params'] as { mean: number[]; cov: number[][] };
    const keep = c['keep'] as number[];
    const out = mvnMarginal(params, keep);
    const expected = c['expected'] as { mean: number[]; cov: number[][] };
    out.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, 9));
    out.cov.forEach((row, i) =>
      row.forEach((v, j) => expect(v).toBeCloseTo(expected.cov[i]![j]!, 9)),
    );
  });
});

describe('mvnConditional', () => {
  it('matches the closed form (PRML 2.81-2.82) when conditioning on a single coordinate', () => {
    const c = casesFor('mvnConditional')[0]!;
    const params = c['params'] as { mean: number[]; cov: number[][] };
    const observed = new Map(
      Object.entries(c['observed'] as Record<string, number>).map(([k, v]) => [Number(k), v]),
    );
    const out = mvnConditional(params, observed);
    const expected = c['expected'] as { mean: number[]; cov: number[][] };
    out.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, 9));
    out.cov.forEach((row, i) =>
      row.forEach((v, j) => expect(v).toBeCloseTo(expected.cov[i]![j]!, 9)),
    );
  });

  it('matches the closed form when conditioning on two coordinates, returning coordinates in ascending index order', () => {
    const c = casesFor('mvnConditional')[1]!;
    const params = c['params'] as { mean: number[]; cov: number[][] };
    const observed = new Map(
      Object.entries(c['observed'] as Record<string, number>).map(([k, v]) => [Number(k), v]),
    );
    const out = mvnConditional(params, observed);
    const expected = c['expected'] as { mean: number[]; cov: number[][] };
    expect(out.mean).toHaveLength(1);
    out.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, 9));
    out.cov.forEach((row, i) =>
      row.forEach((v, j) => expect(v).toBeCloseTo(expected.cov[i]![j]!, 9)),
    );
  });
});

describe('mvnCovarianceEllipse', () => {
  it('for a unit isotropic Gaussian at mass=0.95, the radius matches scipy chi2.ppf(0.95, df=2)', () => {
    const c = casesFor('mvnCovarianceEllipse_chi2Quantile').find((c) => c['mass'] === 0.95)!;
    const expectedRadius = Math.sqrt(c['expected'] as number);
    const ellipse = mvnCovarianceEllipse(
      {
        mean: [0, 0],
        cov: [
          [1, 0],
          [0, 1],
        ],
      },
      0.95,
    );
    expect(ellipse.cx).toBeCloseTo(0, 12);
    expect(ellipse.cy).toBeCloseTo(0, 12);
    expect(ellipse.rx).toBeCloseTo(expectedRadius, 9);
    expect(ellipse.ry).toBeCloseTo(expectedRadius, 9);
  });

  it('for a non-diagonal 2-D covariance, every point on the parametric ellipse boundary has Mahalanobis distance^2 = chi2.ppf(mass, 2), independent of any eigenvector sign/labelling convention', () => {
    const mean: [number, number] = [3, -2];
    const cov = [
      [2, 0.8],
      [0.8, 1],
    ];
    const mass = 0.9;
    const chi2q = casesFor('mvnCovarianceEllipse_chi2Quantile').find((c) => c['mass'] === 0.9)![
      'expected'
    ] as number;
    const ellipse = mvnCovarianceEllipse({ mean, cov }, mass);

    // Major axis convention: rx corresponds to the larger eigenvalue.
    expect(ellipse.rx).toBeGreaterThanOrEqual(ellipse.ry);

    const covInv = invert2x2(cov as [[number, number], [number, number]]);
    for (let k = 0; k < 16; k++) {
      const t = (2 * Math.PI * k) / 16;
      const localX = ellipse.rx * Math.cos(t);
      const localY = ellipse.ry * Math.sin(t);
      const cosA = Math.cos(ellipse.angle);
      const sinA = Math.sin(ellipse.angle);
      const px = ellipse.cx + localX * cosA - localY * sinA;
      const py = ellipse.cy + localX * sinA + localY * cosA;
      const dx = px - mean[0];
      const dy = py - mean[1];
      const mahalanobisSq =
        dx * dx * covInv[0][0] + 2 * dx * dy * covInv[0][1] + dy * dy * covInv[1][1];
      expect(mahalanobisSq).toBeCloseTo(chi2q, 6);
    }
  });
});

describe('mvnSample', () => {
  it('matches the true mean and covariance of a known non-diagonal 3-D Gaussian within a standard-error-derived tolerance', () => {
    // No fixture: there is no single correct sample sequence, only a distribution to
    // match, checked here against these input parameters directly (README convention
    // for every sampler in this package).
    const mean = [1.0, -1.0, 2.0];
    const cov = [
      [3.0, 1.0, 0.5],
      [1.0, 2.0, 0.3],
      [0.5, 0.3, 1.0],
    ];
    const n = 200_000;
    const rng = pcg32(2024, 7);
    const samples: number[][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = mvnSample(rng, { mean, cov });

    const d = mean.length;
    const empMean = new Array(d).fill(0);
    for (const s of samples) for (let i = 0; i < d; i++) empMean[i] += s[i]! / n;

    const empCov: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
    for (const s of samples) {
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          empCov[i]![j]! += ((s[i]! - empMean[i]) * (s[j]! - empMean[j])) / n;
        }
      }
    }

    // SE(mean_i) = sqrt(cov_ii / n). A 6-sigma band keeps the false-failure rate
    // negligible while still catching a wrong scale or a transposed Cholesky factor.
    for (let i = 0; i < d; i++) {
      const se = Math.sqrt(cov[i]![i]! / n);
      expect(Math.abs(empMean[i] - mean[i]!)).toBeLessThan(6 * se);
    }

    // Asymptotic variance of a sample-covariance entry under normality is
    // approximately (cov_ii*cov_jj + cov_ij^2) / n (Wishart-based large-n approximation).
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        const se = Math.sqrt((cov[i]![i]! * cov[j]![j]! + cov[i]![j]! ** 2) / n);
        expect(Math.abs(empCov[i]![j]! - cov[i]![j]!)).toBeLessThan(6 * se);
      }
    }
  });
});

function invert2x2(m: [[number, number], [number, number]]): [[number, number], [number, number]] {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  return [
    [m[1][1] / det, -m[0][1] / det],
    [-m[1][0] / det, m[0][0] / det],
  ];
}
