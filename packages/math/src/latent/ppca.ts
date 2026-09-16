import type { Mat, Vec } from '../types.js';
import { mvnLogPdf } from '../distributions/mvn.js';
import { eigSym, inverse } from '../linalg/decompose.js';
import { dot, matmul, matvec, transpose, vecSub } from '../linalg/core.js';
import { covarianceMatrix, dataMean } from './pca.js';

export interface PpcaParams {
  readonly mean: Vec;
  /** D x M: column `m` is the loading of latent dimension `m` onto data space. */
  readonly w: Mat;
  readonly sigma2: number;
}

/** PRML 12.36: `C = W W^T + sigma^2 I`. */
export function ppcaMarginalCov(w: Mat, sigma2: number): Mat {
  const wwT = matmul(w, transpose(w));
  return wwT.map((row, i) => row.map((v, j) => v + (i === j ? sigma2 : 0)));
}

/** PRML 12.31-12.36: the marginal `p(x)` after integrating out the latent variable. */
export function ppcaMarginalLogPdf(x: Vec, params: PpcaParams): number {
  const cov = ppcaMarginalCov(params.w, params.sigma2);
  return mvnLogPdf(x, { mean: params.mean, cov });
}

export function ppcaLogLikelihood(data: Mat, params: PpcaParams): number {
  let total = 0;
  for (const x of data) total += ppcaMarginalLogPdf(x, params);
  return total;
}

function latentM(w: Mat, sigma2: number): Mat {
  const wtw = matmul(transpose(w), w);
  return wtw.map((row, i) => row.map((v, j) => v + (i === j ? sigma2 : 0)));
}

export interface LatentPosterior {
  readonly mean: Vec;
  readonly cov: Mat;
}

/** PRML 12.41-12.42: `p(z|x)`, whose covariance is the same for every `x`. */
export function ppcaLatentPosterior(x: Vec, params: PpcaParams): LatentPosterior {
  const { mean, w, sigma2 } = params;
  const mInv = inverse(latentM(w, sigma2));
  const diff = vecSub(x, mean);
  const postMean = matvec(mInv, matvec(transpose(w), diff));
  const postCov = mInv.map((row) => row.map((v) => v * sigma2));
  return { mean: postMean, cov: postCov };
}

function meanOf(xs: readonly number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}

/**
 * PRML 12.45-12.46, the closed-form maximum-likelihood solution. `W` is identifiable
 * only up to right-multiplication by an arbitrary `M x M` orthogonal matrix `R`
 * (PRML 12.39-12.41: the marginal depends on `W` only through `W W^T`); this fixes
 * `R = I`, i.e. `W`'s columns are the top-`M` eigenvectors scaled by
 * `sqrt(lambda_i - sigma^2)`, so that two calls on the same data are comparable and a
 * caller wanting a different rotation composes one on afterwards.
 */
export function ppcaMLE(data: Mat, latentDim: number): PpcaParams {
  const mean = dataMean(data);
  const cov = covarianceMatrix(data, mean);
  const { values, vectors } = eigSym(cov);
  const d = mean.length;
  const m = latentDim;
  const sigma2 = meanOf(values.slice(m));

  const w: number[][] = Array.from({ length: d }, () => new Array(m).fill(0));
  for (let k = 0; k < m; k++) {
    const scale = Math.sqrt(Math.max(values[k]! - sigma2, 0));
    for (let row = 0; row < d; row++) w[row]![k] = vectors[k]![row]! * scale;
  }
  return { mean, w, sigma2 };
}

export interface PpcaEStepResult {
  readonly ez: Mat;
  readonly ezz: readonly Mat[];
}

/** PRML 12.54-12.55: the E step, `E[z_n]` and `E[z_n z_n^T]` for every point. */
export function ppcaEStep(data: Mat, params: PpcaParams): PpcaEStepResult {
  const { mean, w, sigma2 } = params;
  const mInv = inverse(latentM(w, sigma2));
  const wT = transpose(w);
  const ez: number[][] = [];
  const ezz: Mat[] = [];
  for (const x of data) {
    const zn = matvec(mInv, matvec(wT, vecSub(x, mean)));
    ez.push(zn);
    ezz.push(mInv.map((row, i) => row.map((v, j) => v * sigma2 + zn[i]! * zn[j]!)));
  }
  return { ez, ezz };
}

export interface PpcaMStepResult {
  readonly w: Mat;
  readonly sigma2: number;
}

/** PRML 12.56-12.57: the M step, given fixed `ez`/`ezz` from `ppcaEStep`. */
export function ppcaMStep(data: Mat, mean: Vec, ez: Mat, ezz: readonly Mat[]): PpcaMStepResult {
  const n = data.length;
  const d = mean.length;
  const m = ez[0]?.length ?? 0;

  const sumXz: number[][] = Array.from({ length: d }, () => new Array(m).fill(0));
  data.forEach((x, i) => {
    const diff = vecSub(x, mean);
    const zn = ez[i]!;
    for (let a = 0; a < d; a++) for (let b = 0; b < m; b++) sumXz[a]![b]! += diff[a]! * zn[b]!;
  });

  const sumZz: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (const e of ezz) for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) sumZz[a]![b]! += e[a]![b]!;

  const w = matmul(sumXz, inverse(sumZz));
  const wt = transpose(w);
  const wtw = matmul(wt, w);

  let total = 0;
  data.forEach((x, i) => {
    const diff = vecSub(x, mean);
    const zn = ez[i]!;
    total += dot(diff, diff);
    total -= 2 * dot(zn, matvec(wt, diff));
    const e = ezz[i]!;
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) total += e[a]![b]! * wtw[a]![b]!;
  });

  return { w, sigma2: total / (n * d) };
}

export interface PpcaEmStepResult {
  readonly ez: Mat;
  readonly params: PpcaParams;
  /** The marginal log-likelihood of `params` entering this step, before the M step moves it. */
  readonly logLikelihood: number;
}

export function ppcaEmStep(data: Mat, params: PpcaParams): PpcaEmStepResult {
  const { ez, ezz } = ppcaEStep(data, params);
  const logLikelihood = ppcaLogLikelihood(data, params);
  const { w, sigma2 } = ppcaMStep(data, params.mean, ez, ezz);
  return { ez, params: { mean: params.mean, w, sigma2 }, logLikelihood };
}

export interface PpcaEmFitResult {
  readonly paramsHistory: readonly PpcaParams[];
  readonly logLikelihoodHistory: readonly number[];
}

/** Mirrors `gmmFitEM`: `paramsHistory[0]` is `initialParams`, one log-likelihood reading per completed E step. */
export function ppcaFitEM(data: Mat, initialParams: PpcaParams, maxIters: number): PpcaEmFitResult {
  const paramsHistory: PpcaParams[] = [initialParams];
  const logLikelihoodHistory: number[] = [];
  let params = initialParams;
  for (let i = 0; i < maxIters; i++) {
    const step = ppcaEmStep(data, params);
    logLikelihoodHistory.push(step.logLikelihood);
    paramsHistory.push(step.params);
    params = step.params;
  }
  return { paramsHistory, logLikelihoodHistory };
}
