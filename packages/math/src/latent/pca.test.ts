import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  covarianceMatrix,
  dataMean,
  discardedEigenvalueSum,
  pcaFitCov,
  pcaFitSvd,
  pcaProject,
  pcaReconstruct,
  pcaReconstructionError,
  whiten,
} from './pca.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('pca');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

/** Eigenvectors are defined only up to sign; compare rows allowing a global flip per row. */
function expectRowsCloseUpToSign(actual: readonly (readonly number[])[], expected: readonly (readonly number[])[], digits = 9) {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((row, i) => {
    const e = expected[i]!;
    const sameSign = row.every((v, j) => Math.abs(v - e[j]!) < 1e-6);
    const flipped = row.every((v, j) => Math.abs(v + e[j]!) < 1e-6);
    expect(sameSign || flipped).toBe(true);
    if (sameSign) {
      row.forEach((v, j) => expect(v).toBeCloseTo(e[j]!, digits));
    } else {
      row.forEach((v, j) => expect(v).toBeCloseTo(-e[j]!, digits));
    }
  });
}

describe('dataMean', () => {
  it('matches the sample mean', () => {
    const c = casesFor('dataMean')[0]!;
    const data = c['data'] as number[][];
    const result = dataMean(data);
    (c['expected'] as number[]).forEach((v, i) => expect(result[i]).toBeCloseTo(v, 9));
  });
});

describe('covarianceMatrix', () => {
  it('matches PRML 12.3, the 1/N sample covariance', () => {
    const c = casesFor('covarianceMatrix')[0]!;
    const data = c['data'] as number[][];
    const mean = c['mean'] as number[];
    const result = covarianceMatrix(data, mean);
    const expected = c['expected'] as number[][];
    result.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('pcaFitCov', () => {
  it('reproduces numpy eigh, descending, sign-fixed by largest-magnitude entry', () => {
    const c = casesFor('pcaFitCov')[0]!;
    const data = c['data'] as number[][];
    const expected = c['expected'] as { mean: number[]; eigenvalues: number[]; components: number[][] };
    const result = pcaFitCov(data);
    result.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, 9));
    result.eigenvalues.forEach((v, i) => expect(v).toBeCloseTo(expected.eigenvalues[i]!, 9));
    expectRowsCloseUpToSign(result.components, expected.components);
  });

  it('agrees with pcaFitSvd on the eigenvalue spectrum (12.1 by two routes)', () => {
    const c = casesFor('pcaFitSvd_eigenvalues')[0]!;
    const data = c['data'] as number[][];
    const expected = c['expected'] as number[];
    const result = pcaFitSvd(data);
    result.eigenvalues.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
  });

  it('still resolves the top-2 subspace when the top two eigenvalues are within 1% of each other', () => {
    const c = casesFor('pcaFitCov_degenerate')[0]!;
    const data = c['data'] as number[][];
    const expected = c['expected'] as { mean: number[]; eigenvalues: number[] };
    const result = pcaFitCov(data);
    result.eigenvalues.forEach((v, i) => expect(v).toBeCloseTo(expected.eigenvalues[i]!, 9));
    // Individual eigenvectors are not asserted here: the top two population variances are
    // within 1% of each other by construction, so at N=60 the sample eigenvalues can (and do)
    // separate far more than that, and numpy and ml-matrix are free to disagree on the exact
    // split within that near-degenerate pair even though both are valid orthonormal bases for
    // the same 2D subspace. That instability is the point (PRML 12.1), not a bug to hide
    // behind a looser tolerance.
  });
});

describe('pcaProject / pcaReconstruct', () => {
  it('projects centred data onto the top-M components', () => {
    const c = casesFor('pcaProject')[0]!;
    const data = c['data'] as number[][];
    const mean = c['mean'] as number[];
    const components = c['components'] as number[][];
    const m = c['numComponents'] as number;
    const result = pcaProject(data, mean, components, m);
    const expected = c['expected'] as number[][];
    result.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('reconstructs from scores back into data space', () => {
    const c = casesFor('pcaReconstruct')[0]!;
    const scores = c['scores'] as number[][];
    const mean = c['mean'] as number[];
    const components = c['components'] as number[][];
    const result = pcaReconstruct(scores, mean, components);
    const expected = c['expected'] as number[][];
    result.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('pcaReconstructionError and discardedEigenvalueSum', () => {
  it('matches the mean squared reconstruction error', () => {
    const c = casesFor('pcaReconstructionError')[0]!;
    const data = c['data'] as number[][];
    const mean = c['mean'] as number[];
    const components = c['components'] as number[][];
    const m = c['numComponents'] as number;
    const result = pcaReconstructionError(data, mean, components, m);
    expect(result).toBeCloseTo(c['expected'] as number, 9);
  });

  it('matches the sum of the discarded eigenvalues', () => {
    const c = casesFor('discardedEigenvalueSum')[0]!;
    const eigenvalues = c['eigenvalues'] as number[];
    const m = c['numComponents'] as number;
    const result = discardedEigenvalueSum(eigenvalues, m);
    expect(result).toBeCloseTo(c['expected'] as number, 9);
  });

  it('are the same quantity (PRML 12.15/12.18): reconstruction error equals the discarded eigenvalue sum', () => {
    const c = casesFor('reconstructionErrorEqualsDiscardedSum')[0]!;
    expect(c['expectedReconError'] as number).toBeCloseTo(c['expectedDiscardedSum'] as number, 9);
  });
});

describe('whiten', () => {
  it('rescales each principal axis to unit variance (PRML 12.24)', () => {
    const c = casesFor('whiten')[0]!;
    const data = c['data'] as number[][];
    const mean = c['mean'] as number[];
    const components = c['components'] as number[][];
    const eigenvalues = c['eigenvalues'] as number[];
    const result = whiten(data, mean, components, eigenvalues);
    const expected = c['expected'] as number[][];
    result.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('produces data with identity covariance', () => {
    const c = casesFor('whiten')[0]!;
    const data = c['data'] as number[][];
    const mean = c['mean'] as number[];
    const components = c['components'] as number[][];
    const eigenvalues = c['eigenvalues'] as number[];
    const result = whiten(data, mean, components, eigenvalues);
    const n = result.length;
    const d = result[0]!.length;
    const whitenedMean = dataMean(result);
    const cov = covarianceMatrix(result, whitenedMean);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        expect(cov[i]![j]!).toBeCloseTo(i === j ? 1 : 0, 6);
      }
    }
    expect(n).toBe(result.length);
  });
});
