import type { Mat, Vec } from '../types.js';
import { symmetrise } from '../linalg/core.js';
import {
  hmmBackwardScaled,
  hmmForwardScaled,
  hmmGammaScaled,
  hmmLogLikelihoodScaled,
  hmmXiScaled,
} from './forwardBackward.js';
import { hmmGaussianEmissionMatrix, type HmmGaussianComponent, type HmmGaussianParams } from './hmm.js';

export type { HmmGaussianComponent, HmmGaussianParams };

export interface HmmEStepResult {
  readonly gamma: number[][];
  readonly xi: number[][][];
  readonly logLikelihood: number;
}

/**
 * The E step of HMM-EM (PRML 13.2.1-13.2.2). Uses the scaled recursion throughout, never
 * the unscaled one, because a real training sequence is exactly the "100 or so" steps the
 * book warns underflows it.
 */
export function hmmEStep(data: Mat, params: HmmGaussianParams): HmmEStepResult {
  const B = hmmGaussianEmissionMatrix(data, params.components);
  const { alphaHat, c } = hmmForwardScaled(params.pi, params.A, B);
  const betaHat = hmmBackwardScaled(params.A, B, c);
  const gamma = hmmGammaScaled(alphaHat, betaHat);
  const xi = hmmXiScaled(alphaHat, params.A, B, betaHat, c);
  const logLikelihood = hmmLogLikelihoodScaled(c);
  return { gamma, xi, logLikelihood };
}

function weightedMeanAndCov(data: Mat, weights: Vec): { mean: number[]; cov: number[][] } {
  const dim = data[0]?.length ?? 0;
  const nk = weights.reduce((a, b) => a + b, 0);
  const mean = new Array(dim).fill(0);
  data.forEach((x, n) => {
    const w = weights[n]!;
    for (let d = 0; d < dim; d++) mean[d] += w * x[d]!;
  });
  for (let d = 0; d < dim; d++) mean[d] /= nk;

  const cov: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  data.forEach((x, n) => {
    const w = weights[n]!;
    for (let a = 0; a < dim; a++) {
      const da = x[a]! - mean[a]!;
      for (let b = 0; b < dim; b++) cov[a]![b]! += w * da * (x[b]! - mean[b]!);
    }
  });
  for (let a = 0; a < dim; a++) for (let b = 0; b < dim; b++) cov[a]![b]! /= nk;
  return { mean, cov: symmetrise(cov) };
}

/**
 * The M step of HMM-EM (PRML 13.18-13.21): each emission component's mean and covariance
 * uses the same `gamma`-weighted formula as chapter 9's mixture M-step, because the book
 * is explicit that `gamma(znk)` plays the role of a responsibility here too (discussion
 * after 13.19). Duplicated locally rather than imported from `mixtures/gmm.ts`, since that
 * module's weighted update is bundled with a mixing-coefficient computation this model
 * does not use (a chain has `pi` and `A`, not per-component weights).
 */
export function hmmMStep(data: Mat, gamma: Mat, xi: readonly Mat[]): HmmGaussianParams {
  const k = gamma[0]?.length ?? 0;

  const pi = gamma[0]!.map((v) => v / gamma[0]!.reduce((a, b) => a + b, 0));

  const xiSum: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  for (const mat of xi) {
    for (let j = 0; j < k; j++) for (let kk = 0; kk < k; kk++) xiSum[j]![kk]! += mat[j]![kk]!;
  }
  const A = xiSum.map((row) => {
    const total = row.reduce((a, b) => a + b, 0);
    return row.map((v) => v / total);
  });

  const components: HmmGaussianComponent[] = [];
  for (let kk = 0; kk < k; kk++) {
    const weights = gamma.map((row) => row[kk]!);
    components.push(weightedMeanAndCov(data, weights));
  }

  return { pi, A, components };
}

export interface HmmEmStepResult {
  readonly gamma: number[][];
  readonly xi: number[][][];
  readonly logLikelihood: number;
  readonly params: HmmGaussianParams;
}

export function hmmEmStep(data: Mat, params: HmmGaussianParams): HmmEmStepResult {
  const { gamma, xi, logLikelihood } = hmmEStep(data, params);
  const nextParams = hmmMStep(data, gamma, xi);
  return { gamma, xi, logLikelihood, params: nextParams };
}

export interface HmmEmFitResult {
  readonly paramsHistory: readonly HmmGaussianParams[];
  readonly gammaHistory: readonly (readonly number[][])[];
  readonly logLikelihoodHistory: readonly number[];
}

// invariant: paramsHistory[0] is initialParams, so it has one more entry than logLikelihoodHistory.
export function hmmFitEM(data: Mat, initialParams: HmmGaussianParams, maxIters: number): HmmEmFitResult {
  const paramsHistory: HmmGaussianParams[] = [initialParams];
  const gammaHistory: number[][][] = [];
  const logLikelihoodHistory: number[] = [];
  let params = initialParams;
  for (let i = 0; i < maxIters; i++) {
    const step = hmmEmStep(data, params);
    gammaHistory.push(step.gamma);
    logLikelihoodHistory.push(step.logLikelihood);
    paramsHistory.push(step.params);
    params = step.params;
  }
  return { paramsHistory, gammaHistory, logLikelihoodHistory };
}
