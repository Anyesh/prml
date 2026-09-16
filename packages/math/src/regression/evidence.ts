import { dot, eigSym, logDet, matmul, matvec, transpose, vecSub } from '../linalg/index.js';
import type { Mat, Vec } from '../types.js';
import type { Hyperparameters } from './bayesianLinear.js';
import { weightPosterior } from './bayesianLinear.js';

/**
 * Log marginal likelihood, PRML 3.86. Returns the log directly because the evidence for a
 * degree-9 fit underflows to zero in double precision well before its log does, and the
 * model-comparison figure plots exactly that range.
 */
export function logEvidence(design: Mat, targets: Vec, hyper: Hyperparameters): number {
  const { alpha, beta } = hyper;
  const n = design.length;
  const m = design[0]?.length ?? 0;
  const { mean, precision } = weightPosterior(design, targets, hyper);
  const residual = vecSub(matvec(design, mean), targets);
  const eMn = 0.5 * beta * dot(residual, residual) + 0.5 * alpha * dot(mean, mean);
  return (
    0.5 * m * Math.log(alpha) +
    0.5 * n * Math.log(beta) -
    eMn -
    0.5 * logDet(precision) -
    0.5 * n * Math.log(2 * Math.PI)
  );
}

export interface EvidenceMaximisation extends Hyperparameters {
  /** `gamma`, PRML 3.91: the number of weight directions the data actually determines. */
  readonly effectiveParameters: number;
  readonly iterations: number;
  readonly converged: boolean;
  readonly logEvidence: number;
}

/**
 * Re-estimates `alpha` and `beta` by the fixed-point iteration of PRML 3.92 and 3.95.
 *
 * Implicit, not a gradient step: each update solves for the eigenvalues of `beta ΦᵀΦ`,
 * which change as `beta` changes, so the recursion must recompute them per iteration.
 * Stops when both hyperparameters move by less than `tol` in relative terms.
 */
export function maximiseEvidence(
  design: Mat,
  targets: Vec,
  initial: Hyperparameters,
  maxIterations = 200,
  tol = 1e-9,
): EvidenceMaximisation {
  const n = design.length;
  const gramEigenvalues = eigSym(matmul(transpose(design), design)).values;

  let alpha = initial.alpha;
  let beta = initial.beta;
  let converged = false;
  let iterations = 0;

  for (let it = 1; it <= maxIterations; it++) {
    iterations = it;
    const { mean } = weightPosterior(design, targets, { alpha, beta });
    const lambdas = gramEigenvalues.map((e) => beta * e);
    const gamma = lambdas.reduce((s, lam) => s + lam / (alpha + lam), 0);
    const newAlpha = gamma / dot(mean, mean);
    const residual = vecSub(matvec(design, mean), targets);
    const newBeta = (n - gamma) / dot(residual, residual);
    converged =
      Math.abs(newAlpha - alpha) <= tol * Math.abs(alpha) &&
      Math.abs(newBeta - beta) <= tol * Math.abs(beta);
    alpha = newAlpha;
    beta = newBeta;
    if (converged) break;
  }

  const finalLambdas = gramEigenvalues.map((e) => beta * e);
  const effectiveParameters = finalLambdas.reduce((s, lam) => s + lam / (alpha + lam), 0);

  return {
    alpha,
    beta,
    effectiveParameters,
    iterations,
    converged,
    logEvidence: logEvidence(design, targets, { alpha, beta }),
  };
}
