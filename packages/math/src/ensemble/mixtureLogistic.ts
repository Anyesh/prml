import { dot } from '../linalg/index.js';
import { logSumExp, sigmoid } from '../numeric.js';
import { weightedLogisticFit } from './weightedLogistic.js';
import type { Mat, Vec } from '../types.js';

export interface LogisticMixtureParams {
  readonly weights: Mat;
  readonly mixing: Vec;
}

function logBernoulliTerm(y: number, target: number): number {
  return target * Math.log(y) + (1 - target) * Math.log(1 - y);
}

/**
 * `ln(pi_k) + ln(y_k^t (1-y_k)^(1-t))`, PRML 14.46's per-point per-component term, kept in
 * log form throughout rather than multiplying the raw `y_k^t(1-y_k)^(1-t)` across the K
 * components before taking a single logarithm: a component whose fit saturates `y_k` toward
 * 0 or 1 still contributes a finite term here, where the naive product would already have
 * rounded to exactly 0 and poisoned both the responsibility ratio and the log-likelihood.
 */
function logWeightedComponentTerms(design: Mat, targets: Vec, params: LogisticMixtureParams): number[][] {
  return design.map((row, n) =>
    params.weights.map((w, k) => Math.log(params.mixing[k]!) + logBernoulliTerm(sigmoid(dot(w, row)), targets[n]!)),
  );
}

/** PRML 14.48: responsibilities as a softmax of the log-weighted component terms, one row per point. */
export function mixtureLogisticResponsibilities(design: Mat, targets: Vec, params: LogisticMixtureParams): number[][] {
  return logWeightedComponentTerms(design, targets, params).map((row) => {
    const lse = logSumExp(row);
    return row.map((v) => Math.exp(v - lse));
  });
}

/** PRML 14.46, via `logSumExp` over the K per-component log terms at each point, summed over the dataset. */
export function mixtureLogisticLogLikelihood(design: Mat, targets: Vec, params: LogisticMixtureParams): number {
  let total = 0;
  for (const row of logWeightedComponentTerms(design, targets, params)) total += logSumExp(row);
  return total;
}

/**
 * PRML 14.50 (mixing coefficients, the responsibility column mean) plus one
 * `weightedLogisticFit` per component (14.51-14.52). The M step decouples across components
 * exactly the way `weightedLogisticFit`'s own doc comment describes: `pointWeights` is that
 * component's responsibility column and `targets` stays the true 0/1 label throughout, since
 * only the per-point weight on the cross-entropy term differs between components, not the
 * label being fit.
 */
export function mixtureLogisticMStep(design: Mat, targets: Vec, responsibilities: number[][]): LogisticMixtureParams {
  const n = design.length;
  const k = responsibilities[0]?.length ?? 0;

  const mixing = new Array(k).fill(0);
  for (const row of responsibilities) for (let c = 0; c < k; c++) mixing[c]! += row[c]!;
  for (let c = 0; c < k; c++) mixing[c]! /= n;

  const weights = Array.from({ length: k }, (_, c) => {
    const pointWeights = responsibilities.map((row) => row[c]!);
    return weightedLogisticFit(design, targets, pointWeights).weights;
  });

  return { weights, mixing };
}

export interface MixtureLogisticFit {
  readonly paramsHistory: readonly LogisticMixtureParams[];
  readonly responsibilitiesHistory: readonly (number[][])[];
  readonly logLikelihoodHistory: readonly number[];
}

/**
 * Mirrors `mixtures/em.ts`'s `gmmFitEM`: `logLikelihoodHistory[i]` and
 * `responsibilitiesHistory[i]` are both evaluated at `paramsHistory[i]` (the mixture
 * entering iterate `i`), before that iterate's M step moves it, so `paramsHistory` ends up
 * one entry longer than the two history arrays.
 */
export function mixtureLogisticFitEM(
  design: Mat,
  targets: Vec,
  initialParams: LogisticMixtureParams,
  maxIters: number,
): MixtureLogisticFit {
  const paramsHistory: LogisticMixtureParams[] = [initialParams];
  const responsibilitiesHistory: number[][][] = [];
  const logLikelihoodHistory: number[] = [];
  let params = initialParams;
  for (let i = 0; i < maxIters; i++) {
    const responsibilities = mixtureLogisticResponsibilities(design, targets, params);
    logLikelihoodHistory.push(mixtureLogisticLogLikelihood(design, targets, params));
    responsibilitiesHistory.push(responsibilities);
    params = mixtureLogisticMStep(design, targets, responsibilities);
    paramsHistory.push(params);
  }
  return { paramsHistory, responsibilitiesHistory, logLikelihoodHistory };
}
