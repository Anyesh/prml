import { dot } from '../linalg/index.js';
import { logSumExp, softmax } from '../numeric.js';
import { weightedLeastSquaresFit } from './weightedLeastSquares.js';
import { weightedSoftmaxFit } from './weightedSoftmax.js';
import type { Mat, Vec } from '../types.js';

export interface ExpertsParams {
  readonly expertWeights: Mat; // K x D_expert
  readonly beta: number; // shared noise precision across experts
  readonly gateWeights: Mat; // K x D_gate, softmax gate; row 0 is conventionally zero once fitted, see weightedSoftmax.ts
}

function gaussianLogPdf(target: number, mean: number, beta: number): number {
  return -0.5 * Math.log(2 * Math.PI) + 0.5 * Math.log(beta) - 0.5 * beta * (target - mean) ** 2;
}

/** `pi(x) = softmax(V phi_gate(x))`, PRML 4.104-4.105's K-class softmax gate. */
function gateProbabilities(designGateRow: Vec, gateWeights: Mat): number[] {
  return softmax(gateWeights.map((w) => dot(w, designGateRow)));
}

/**
 * `ln(pi_k(x_n)) + ln N(t_n|w_k^T phi_n, beta^-1)`, PRML 14.53's numerator terms in log
 * form, one row per point and one column per expert. Kept in log form throughout so that
 * a point far into one expert's tail does not underflow its density to exactly 0 before
 * the mixture sum or the responsibility ratio sees it.
 */
function logWeightedExpertTerms(designExperts: Mat, designGate: Mat, targets: Vec, params: ExpertsParams): number[][] {
  return designExperts.map((row, n) => {
    const pis = gateProbabilities(designGate[n]!, params.gateWeights);
    return params.expertWeights.map((w, k) => Math.log(pis[k]!) + gaussianLogPdf(targets[n]!, dot(w, row), params.beta));
  });
}

/** PRML 14.53: a softmax of the K log-weighted expert terms, one row per point. */
export function mixtureExpertsResponsibilities(designExperts: Mat, designGate: Mat, targets: Vec, params: ExpertsParams): number[][] {
  return logWeightedExpertTerms(designExperts, designGate, targets, params).map((row) => {
    const lse = logSumExp(row);
    return row.map((v) => Math.exp(v - lse));
  });
}

/** `sum_n ln( sum_k pi_k(x_n) N(t_n|w_k^T phi_n,beta^-1) )`, via `logSumExp` of the K log terms per point. */
export function mixtureExpertsLogLikelihood(designExperts: Mat, designGate: Mat, targets: Vec, params: ExpertsParams): number {
  let total = 0;
  for (const row of logWeightedExpertTerms(designExperts, designGate, targets, params)) total += logSumExp(row);
  return total;
}

/**
 * Expert weights: `weightedLeastSquaresFit` per component with weight = that component's
 * responsibility column (PRML 14.42's mixture-of-experts specialisation, one call per
 * expert regardless of K). Beta: the pooled precision update in the same form as PRML
 * 14.44 but with input-dependent (gated) responsibilities in place of a fixed mixing
 * coefficient; because each row of `responsibilities` sums to 1, `sum_n sum_k gamma_nk`
 * is exactly `N`, so the pooled sum of squared residuals is divided by the point count
 * directly.
 *
 * Gate weights: maximising `sum_n sum_k gamma_nk ln pi_k(x_n)` over the gate parameters
 * `V` is exactly `weightedSoftmaxFit`'s own objective with `targets = responsibilities`
 * (each row already sums to 1, exactly the shape `weightedSoftmaxFit` expects) and
 * uniform `pointWeights` of 1 (every point contributes to the gate's cross-entropy once,
 * unlike the responsibility-weighted per-expert fits above). This is the K-class
 * generalisation of 4.106's cross-entropy with a soft target in place of a one-hot label,
 * which is exactly what `weightedSoftmaxFit`'s own doc comment describes.
 */
export function mixtureExpertsMStep(designExperts: Mat, designGate: Mat, targets: Vec, responsibilities: number[][]): ExpertsParams {
  const n = designExperts.length;
  const k = responsibilities[0]?.length ?? 0;

  const expertWeights = Array.from({ length: k }, (_, c) =>
    weightedLeastSquaresFit(
      designExperts,
      targets,
      responsibilities.map((row) => row[c]!),
    ),
  );

  let weightedSquaredError = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      const mean = dot(expertWeights[c]!, designExperts[i]!);
      weightedSquaredError += responsibilities[i]![c]! * (targets[i]! - mean) ** 2;
    }
  }
  const beta = n / weightedSquaredError;

  const gateWeights = weightedSoftmaxFit(designGate, responsibilities, new Array(n).fill(1)).weights;

  return { expertWeights, beta, gateWeights };
}

export interface MixtureExpertsFit {
  readonly paramsHistory: readonly ExpertsParams[];
  readonly responsibilitiesHistory: readonly (number[][])[];
  readonly logLikelihoodHistory: readonly number[];
}

/**
 * Mirrors `mixtures/em.ts`'s `gmmFitEM`: `logLikelihoodHistory[i]` and
 * `responsibilitiesHistory[i]` are both evaluated at `paramsHistory[i]` (the mixture
 * entering iterate `i`), before that iterate's M step moves it, so `paramsHistory` ends up
 * one entry longer than the two history arrays.
 */
export function mixtureExpertsFitEM(
  designExperts: Mat,
  designGate: Mat,
  targets: Vec,
  initialParams: ExpertsParams,
  maxIters: number,
): MixtureExpertsFit {
  const paramsHistory: ExpertsParams[] = [initialParams];
  const responsibilitiesHistory: number[][][] = [];
  const logLikelihoodHistory: number[] = [];
  let params = initialParams;
  for (let i = 0; i < maxIters; i++) {
    const responsibilities = mixtureExpertsResponsibilities(designExperts, designGate, targets, params);
    logLikelihoodHistory.push(mixtureExpertsLogLikelihood(designExperts, designGate, targets, params));
    responsibilitiesHistory.push(responsibilities);
    params = mixtureExpertsMStep(designExperts, designGate, targets, responsibilities);
    paramsHistory.push(params);
  }
  return { paramsHistory, responsibilitiesHistory, logLikelihoodHistory };
}
