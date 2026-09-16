import type { Mat, Vec } from '../types.js';
import { eigSym, svd } from '../linalg/decompose.js';
import { matmul, transpose, vecSub } from '../linalg/core.js';

export function dataMean(data: Mat): Vec {
  const n = data.length;
  const dim = data[0]?.length ?? 0;
  const sum = new Array(dim).fill(0);
  for (const x of data) for (let d = 0; d < dim; d++) sum[d] += x[d]!;
  return sum.map((v) => v / n);
}

/** PRML 12.3: the 1/N sample covariance about a given mean (never re-estimates the mean itself). */
export function covarianceMatrix(data: Mat, mean: Vec): Mat {
  const n = data.length;
  const dim = mean.length;
  const cov: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  for (const x of data) {
    for (let i = 0; i < dim; i++) {
      const di = x[i]! - mean[i]!;
      for (let j = 0; j < dim; j++) cov[i]![j]! += di * (x[j]! - mean[j]!);
    }
  }
  for (let i = 0; i < dim; i++) for (let j = 0; j < dim; j++) cov[i]![j]! /= n;
  return cov;
}

export interface PcaResult {
  readonly mean: Vec;
  /** Eigenvalues of the covariance, descending (PRML 12.5-12.6). */
  readonly eigenvalues: Vec;
  /** Principal axes as rows, unit length, sign-fixed to match `eigSym` (largest-magnitude entry positive). */
  readonly components: Mat;
}

/** PCA by eigendecomposition of the covariance matrix (PRML 12.1.1). */
export function pcaFitCov(data: Mat): PcaResult {
  const mean = dataMean(data);
  const cov = covarianceMatrix(data, mean);
  const { values, vectors } = eigSym(cov);
  return { mean, eigenvalues: values, components: vectors };
}

/**
 * PCA by SVD of the centred data matrix (PRML 12.1.4's route for `N < D`, but valid in
 * general): singular values `s` of the centred `N x D` data relate to the covariance's
 * eigenvalues by `s^2 / N`, and the right singular vectors are the principal axes. Sign
 * is fixed the same way as `eigSym` so the two routes are directly comparable, since SVD
 * itself carries no sign convention (ml-matrix and numpy are free to disagree).
 */
export function pcaFitSvd(data: Mat): PcaResult {
  const n = data.length;
  const mean = dataMean(data);
  const centered = data.map((x) => vecSub(x, mean));
  const { s, v } = svd(centered);
  const eigenvalues = s.map((sv) => (sv * sv) / n);
  const componentsRaw = transpose(v); // rows of v^T = right singular vectors as rows
  const components = componentsRaw.map((row) => {
    let maxIdx = 0;
    for (let i = 1; i < row.length; i++) if (Math.abs(row[i]!) > Math.abs(row[maxIdx]!)) maxIdx = i;
    return row[maxIdx]! < 0 ? row.map((v2) => -v2) : row;
  });
  return { mean, eigenvalues, components };
}

/** Centred data projected onto the first `numComponents` rows of `components` (PRML 12.10's `a_ni` restricted to `i <= M`). */
export function pcaProject(data: Mat, mean: Vec, components: Mat, numComponents: number): Mat {
  const top = components.slice(0, numComponents);
  return data.map((x) => {
    const centered = vecSub(x, mean);
    return top.map((u) => {
      let sum = 0;
      for (let d = 0; d < centered.length; d++) sum += centered[d]! * u[d]!;
      return sum;
    });
  });
}

/** PRML 12.19-12.21: reconstruction from `M`-dimensional scores back into data space. */
export function pcaReconstruct(scores: Mat, mean: Vec, components: Mat): Mat {
  const numComponents = scores[0]?.length ?? 0;
  const top = components.slice(0, numComponents);
  return scores.map((score) => {
    const out = [...mean];
    for (let m = 0; m < numComponents; m++) {
      const u = top[m]!;
      const s = score[m]!;
      for (let d = 0; d < out.length; d++) out[d]! += s * u[d]!;
    }
    return out;
  });
}

/** Mean squared reconstruction error keeping the top `numComponents` axes (PRML 12.15, before the eigenvector substitution). */
export function pcaReconstructionError(data: Mat, mean: Vec, components: Mat, numComponents: number): number {
  const scores = pcaProject(data, mean, components, numComponents);
  const reconstructed = pcaReconstruct(scores, mean, components);
  let total = 0;
  data.forEach((x, n) => {
    const r = reconstructed[n]!;
    for (let d = 0; d < x.length; d++) {
      const diff = x[d]! - r[d]!;
      total += diff * diff;
    }
  });
  return total / data.length;
}

/** PRML 12.18: `J = sum_{i=M+1}^{D} lambda_i`, the discarded-eigenvalue form of the same reconstruction error. */
export function discardedEigenvalueSum(eigenvalues: Vec, numComponents: number): number {
  let sum = 0;
  for (let i = numComponents; i < eigenvalues.length; i++) sum += eigenvalues[i]!;
  return sum;
}

/**
 * PRML 12.24: `y = Lambda^{-1/2} U (x - xbar)`, rescaling each principal axis to unit
 * variance. `components` and `eigenvalues` must already be sign/order-consistent (as
 * returned together by `pcaFitCov`/`pcaFitSvd`), since whitening divides axis `i` by
 * `sqrt(eigenvalues[i])` and a mismatched pairing would rescale the wrong direction.
 */
export function whiten(data: Mat, mean: Vec, components: Mat, eigenvalues: Vec): Mat {
  const projected = matmul(
    data.map((x) => vecSub(x, mean)),
    transpose(components),
  );
  const scaled = transpose(projected).map((col, i) => col.map((v) => v / Math.sqrt(eigenvalues[i]!)));
  return transpose(scaled);
}
