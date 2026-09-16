import type { Mat, Vec } from '../types.js';
import { logSumExp } from '../numeric.js';
import { dot } from '../linalg/core.js';
import { weightedLeastSquaresFit } from './weightedLeastSquares.js';

/** PRML 14.34: `p(t|phi) = sum_k pi_k N(t | w_k^T phi, beta^-1)`, one shared `beta` across K linear-regression means. */
export interface LinearMixtureParams {
  readonly weights: Mat;
  readonly mixing: Vec;
  readonly beta: number;
}

function componentLogPdf(phi: Vec, t: number, w: Vec, beta: number): number {
  const mean = dot(phi, w);
  const residual = t - mean;
  return -0.5 * Math.log(2 * Math.PI) + 0.5 * Math.log(beta) - 0.5 * beta * residual * residual;
}

function logWeightedComponentPdfs(phi: Vec, t: number, params: LinearMixtureParams): number[] {
  return params.weights.map((w, k) => Math.log(params.mixing[k]!) + componentLogPdf(phi, t, w, params.beta));
}

/** PRML 14.37 in log space, via `logSumExp`, so a component far from `(phi, t)` underflows instead of poisoning the ratio. */
export function mixtureLinearRegressionResponsibilities(design: Mat, targets: Vec, params: LinearMixtureParams): number[][] {
  return design.map((phi, n) => {
    const logWeighted = logWeightedComponentPdfs(phi, targets[n]!, params);
    const lse = logSumExp(logWeighted);
    return logWeighted.map((v) => Math.exp(v - lse));
  });
}

/** PRML 14.35: `sum_n ln( sum_k pi_k N(t_n | w_k^T phi_n, beta^-1) )`, again via `logSumExp` for stability. */
export function mixtureLinearRegressionLogLikelihood(design: Mat, targets: Vec, params: LinearMixtureParams): number {
  let total = 0;
  design.forEach((phi, n) => {
    total += logSumExp(logWeightedComponentPdfs(phi, targets[n]!, params));
  });
  return total;
}

/**
 * PRML 14.38 (mixing coefficients), 14.42 (weights, one `weightedLeastSquaresFit` per
 * component with weight = responsibility column `k`) and 14.44 (beta, pooled across all K
 * components). Beta is computed from the `weights` just solved above, not the mixture that
 * produced `responsibilities`: the M step maximises `Q(theta, theta_old)` over the whole
 * new `theta` at once, so every parameter in it must be evaluated against the fully updated
 * set, not a mix of old and new.
 */
export function mixtureLinearRegressionMStep(design: Mat, targets: Vec, responsibilities: number[][]): LinearMixtureParams {
  const n = design.length;
  const k = responsibilities[0]?.length ?? 0;

  const nk = new Array(k).fill(0);
  for (const row of responsibilities) for (let c = 0; c < k; c++) nk[c]! += row[c]!;
  const mixing = nk.map((v) => v / n);

  const weights: number[][] = [];
  for (let c = 0; c < k; c++) {
    const columnWeights = responsibilities.map((row) => row[c]!);
    weights.push(weightedLeastSquaresFit(design, targets, columnWeights));
  }

  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const phi = design[i]!;
    const t = targets[i]!;
    for (let c = 0; c < k; c++) {
      const residual = t - dot(phi, weights[c]!);
      sumSq += responsibilities[i]![c]! * residual * residual;
    }
  }
  const beta = n / sumSq;

  return { weights, mixing, beta };
}

export interface MixtureLinearRegressionFit {
  readonly paramsHistory: readonly LinearMixtureParams[];
  readonly responsibilitiesHistory: readonly (number[][])[];
  readonly logLikelihoodHistory: readonly number[];
}

/**
 * Runs `maxIters` E/M rounds from `initialParams`, deterministically (no RNG).
 * `paramsHistory[0]` is `initialParams`, so `paramsHistory` has `maxIters + 1` entries
 * against `maxIters` log-likelihood readings; `logLikelihoodHistory[i]` is evaluated at
 * `paramsHistory[i]`, before the i-th M step moves it, matching `mixtures/em.ts`'s
 * `gmmFitEM` convention.
 */
export function mixtureLinearRegressionFitEM(
  design: Mat,
  targets: Vec,
  initialParams: LinearMixtureParams,
  maxIters: number,
): MixtureLinearRegressionFit {
  const paramsHistory: LinearMixtureParams[] = [initialParams];
  const responsibilitiesHistory: number[][][] = [];
  const logLikelihoodHistory: number[] = [];
  let params = initialParams;
  for (let i = 0; i < maxIters; i++) {
    const responsibilities = mixtureLinearRegressionResponsibilities(design, targets, params);
    const logLikelihood = mixtureLinearRegressionLogLikelihood(design, targets, params);
    const nextParams = mixtureLinearRegressionMStep(design, targets, responsibilities);
    responsibilitiesHistory.push(responsibilities);
    logLikelihoodHistory.push(logLikelihood);
    paramsHistory.push(nextParams);
    params = nextParams;
  }
  return { paramsHistory, responsibilitiesHistory, logLikelihoodHistory };
}
