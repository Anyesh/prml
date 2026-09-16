import type { Mat, Vec } from '../types.js';
import { logSumExp } from '../numeric.js';
import { mvnLogPdf } from '../distributions/mvn.js';
import { symmetrise } from '../linalg/core.js';

export interface GmmComponent {
  readonly weight: number;
  readonly mean: Vec;
  readonly cov: Mat;
}

export interface GmmParams {
  readonly components: readonly GmmComponent[];
}

export function gmmComponentLogPdf(x: Vec, c: GmmComponent): number {
  return mvnLogPdf(x, { mean: c.mean, cov: c.cov });
}

function logWeightedComponentPdfs(x: Vec, params: GmmParams): number[] {
  return params.components.map((c) => Math.log(c.weight) + gmmComponentLogPdf(x, c));
}

/** PRML 9.7 in log space, via `logSumExp`, so a component far from `x` underflows instead of poisoning the sum. */
export function gmmLogPdf(x: Vec, params: GmmParams): number {
  return logSumExp(logWeightedComponentPdfs(x, params));
}

export function gmmPdf(x: Vec, params: GmmParams): number {
  return Math.exp(gmmLogPdf(x, params));
}

/**
 * PRML 9.13, the E step for one point: `softmax` of the log-weighted component
 * densities, so two components with wildly different densities still divide responsibility
 * correctly instead of the smaller one rounding to exactly zero before the ratio is taken.
 */
export function gmmResponsibilities(x: Vec, params: GmmParams): number[] {
  const logWeighted = logWeightedComponentPdfs(x, params);
  const lse = logSumExp(logWeighted);
  return logWeighted.map((v) => Math.exp(v - lse));
}

/** `gmmResponsibilities` stacked over a dataset: row `n` is the responsibility vector for `data[n]`. */
export function gmmEStep(data: Mat, params: GmmParams): number[][] {
  return data.map((x) => gmmResponsibilities(x, params));
}

export function gmmLogLikelihood(data: Mat, params: GmmParams): number {
  let total = 0;
  for (const x of data) total += gmmLogPdf(x, params);
  return total;
}

function empiricalCovariance(points: Mat, mean: Vec): number[][] {
  const dim = mean.length;
  const cov: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  for (const x of points) {
    for (let i = 0; i < dim; i++) {
      const di = x[i]! - mean[i]!;
      for (let j = 0; j < dim; j++) {
        cov[i]![j]! += di * (x[j]! - mean[j]!);
      }
    }
  }
  const n = points.length;
  return symmetrise(cov.map((row) => row.map((v) => v / n)));
}

/**
 * Builds a mixture from a hard clustering (typically K-means): mixing coefficient as
 * cluster fraction, mean and covariance as the cluster's own empirical moments. This is
 * the initialisation PRML recommends before the first E step, so that EM starts from a
 * sensible partition rather than an arbitrary one.
 */
export function gmmInit(data: Mat, means: Mat, assignments: readonly number[], k: number): GmmParams {
  const n = data.length;
  const components: GmmComponent[] = [];
  for (let c = 0; c < k; c++) {
    const points = data.filter((_, i) => assignments[i] === c);
    components.push({
      weight: points.length / n,
      mean: [...means[c]!],
      cov: empiricalCovariance(points, means[c]!),
    });
  }
  return { components };
}

/**
 * PRML 9.17 (mean), 9.19 (covariance) and 9.22 (mixing coefficient): the closed-form M
 * step given fixed responsibilities. `Nk`, the effective number of points assigned to
 * component `k`, is the responsibility column sum, and can be fractional.
 */
export function gmmMStep(data: Mat, responsibilities: Mat): GmmParams {
  const n = data.length;
  const k = responsibilities[0]?.length ?? 0;
  const dim = data[0]?.length ?? 0;

  const nk = new Array(k).fill(0);
  for (const row of responsibilities) for (let c = 0; c < k; c++) nk[c]! += row[c]!;

  const means: number[][] = Array.from({ length: k }, () => new Array(dim).fill(0));
  for (let i = 0; i < n; i++) {
    const x = data[i]!;
    for (let c = 0; c < k; c++) {
      const r = responsibilities[i]![c]!;
      for (let d = 0; d < dim; d++) means[c]![d]! += r * x[d]!;
    }
  }
  for (let c = 0; c < k; c++) for (let d = 0; d < dim; d++) means[c]![d]! /= nk[c]!;

  const covs: number[][][] = Array.from({ length: k }, () => Array.from({ length: dim }, () => new Array(dim).fill(0)));
  for (let i = 0; i < n; i++) {
    const x = data[i]!;
    for (let c = 0; c < k; c++) {
      const r = responsibilities[i]![c]!;
      const mean = means[c]!;
      for (let a = 0; a < dim; a++) {
        const da = x[a]! - mean[a]!;
        for (let b = 0; b < dim; b++) {
          covs[c]![a]![b]! += r * da * (x[b]! - mean[b]!);
        }
      }
    }
  }
  for (let c = 0; c < k; c++) for (let a = 0; a < dim; a++) for (let b = 0; b < dim; b++) covs[c]![a]![b]! /= nk[c]!;

  const components: GmmComponent[] = Array.from({ length: k }, (_, c) => ({
    weight: nk[c]! / n,
    mean: means[c]!,
    cov: symmetrise(covs[c]!),
  }));
  return { components };
}
