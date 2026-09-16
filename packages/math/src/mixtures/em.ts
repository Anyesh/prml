import type { Mat } from '../types.js';
import { gmmComponentLogPdf, gmmEStep, gmmLogLikelihood, gmmMStep, type GmmParams } from './gmm.js';

/** PRML 9.40, the Q function: the complete-data log-likelihood, expected under fixed responsibilities. */
export function expectedCompleteDataLogLikelihood(data: Mat, responsibilities: Mat, params: GmmParams): number {
  let total = 0;
  data.forEach((x, n) => {
    params.components.forEach((c, k) => {
      const r = responsibilities[n]![k]!;
      if (r === 0) return;
      total += r * (Math.log(c.weight) + gmmComponentLogPdf(x, c));
    });
  });
  return total;
}

/**
 * `-sum r log r`, the entropy of the responsibility distribution at every point, summed
 * over the dataset. A responsibility of exactly 0 contributes 0 by the standard convention
 * `0 log 0 = 0` (the limit of `x log x` as `x -> 0`), so it is skipped rather than
 * evaluated, which would otherwise be `-0 * -Infinity = NaN`.
 */
export function responsibilityEntropy(responsibilities: Mat): number {
  let total = 0;
  for (const row of responsibilities) {
    for (const r of row) {
      if (r === 0) continue;
      total -= r * Math.log(r);
    }
  }
  return total;
}

/**
 * PRML 9.71 specialised to a Gaussian mixture: `L(q, theta) = Q(theta) + H(q)`, the
 * expected complete-data log-likelihood plus the entropy of `q`. This is a lower bound on
 * `gmmLogLikelihood` for any responsibility matrix `q`, with equality exactly when `q` is
 * the true posterior responsibilities for `params` (PRML 9.70, KL = 0 at the E step).
 */
export function emLowerBound(data: Mat, responsibilities: Mat, params: GmmParams): number {
  return expectedCompleteDataLogLikelihood(data, responsibilities, params) + responsibilityEntropy(responsibilities);
}

/** `ln p(X|theta) - L(q, theta)`, the KL divergence between `q` and the true posterior (PRML 9.72). Never negative. */
export function emKlGap(data: Mat, responsibilities: Mat, params: GmmParams): number {
  return gmmLogLikelihood(data, params) - emLowerBound(data, responsibilities, params);
}

export interface EmStepResult {
  readonly responsibilities: Mat;
  readonly params: GmmParams;
  /** The log-likelihood of `params` (the mixture entering this step), before the M step moves it. */
  readonly logLikelihood: number;
}

/**
 * One full EM iterate: an E step against `params`, whose responsibilities tighten the
 * bound to touch the log-likelihood exactly, then an M step that maximises that bound over
 * new parameters. `logLikelihood` is evaluated at the incoming `params`, not the outgoing
 * ones, so a caller plotting a curve gets one log-likelihood value per responsibility set.
 */
export function gmmEmStep(data: Mat, params: GmmParams): EmStepResult {
  const responsibilities = gmmEStep(data, params);
  const logLikelihood = gmmLogLikelihood(data, params);
  const nextParams = gmmMStep(data, responsibilities);
  return { responsibilities, params: nextParams, logLikelihood };
}

export interface EmFitResult {
  readonly paramsHistory: readonly GmmParams[];
  readonly responsibilitiesHistory: readonly Mat[];
  readonly logLikelihoodHistory: readonly number[];
}

/**
 * Runs `gmmEmStep` for `maxIters` rounds from `initialParams`, deterministically (no RNG):
 * the seeded initialisation lives in `kmeansInit` and `gmmInit`, called by the widget
 * before this. `paramsHistory[0]` is `initialParams`, so `paramsHistory` has `maxIters + 1`
 * entries against `maxIters` log-likelihood readings, one per completed E step.
 */
export function gmmFitEM(data: Mat, initialParams: GmmParams, maxIters: number): EmFitResult {
  const paramsHistory: GmmParams[] = [initialParams];
  const responsibilitiesHistory: Mat[] = [];
  const logLikelihoodHistory: number[] = [];
  let params = initialParams;
  for (let i = 0; i < maxIters; i++) {
    const step = gmmEmStep(data, params);
    responsibilitiesHistory.push(step.responsibilities);
    logLikelihoodHistory.push(step.logLikelihood);
    paramsHistory.push(step.params);
    params = step.params;
  }
  return { paramsHistory, responsibilitiesHistory, logLikelihoodHistory };
}
