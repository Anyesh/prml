import { cholesky, dot, matmul, matvec, solveCholesky, symmetrise, transpose, vecSub, zeros } from '../linalg/index.js';
import { sigmoid } from '../numeric.js';
import type { Mat, Vec } from '../types.js';

/**
 * `y(1-y)` underflows to exactly 0 once the sigmoid saturates, same floor as
 * `classification/irls.ts`'s R_FLOOR and for the same reason: a zero here can zero out an
 * entire diagonal entry of the weighted curvature and make the Newton system singular right
 * where a component's fit is most confident.
 */
const CURVATURE_FLOOR = 1e-10;

export interface WeightedLogisticFit {
  readonly weights: number[];
  readonly iterations: number;
  readonly converged: boolean;
}

export interface WeightedLogisticFitOptions {
  readonly initialWeights?: Vec;
  readonly maxIterations?: number;
  readonly tol?: number;
}

/**
 * Newton-Raphson on the per-point-weighted cross-entropy error
 * `sum_n pointWeights[n] * [-targets[n] ln y_n - (1 - targets[n]) ln(1 - y_n)]`,
 * generalising `classification/irls.ts`'s plain IRLS (PRML 4.99-4.100) with a data weight.
 *
 * The gradient and Hessian identities (`sum w_n(y_n - t_n) phi_n`,
 * `sum w_n y_n(1 - y_n) phi_n phi_n^T`) never assumed `targets[n]` in {0, 1}: they hold for
 * any target in [0, 1], which is what lets one function serve both call sites in this
 * chapter. `pointWeights = 1` and `targets` in {0, 1} recovers plain logistic regression;
 * `pointWeights` = a mixture responsibility with binary `targets` is the M-step of the
 * mixture of logistic regression models (14.50-14.52); uniform `pointWeights` with `targets`
 * a soft responsibility in [0, 1] is the gating-network M-step of the mixture of experts
 * (14.5.3), fitting the gate to reproduce the responsibilities EM assigned it.
 */
export function weightedLogisticFit(
  design: Mat,
  targets: Vec,
  pointWeights: Vec,
  options?: WeightedLogisticFitOptions,
): WeightedLogisticFit {
  const dimension = design[0]?.length ?? 0;
  const maxIterations = options?.maxIterations ?? 100;
  const tol = options?.tol ?? 1e-12;
  let weights = options?.initialWeights ? [...options.initialWeights] : zeros(dimension);

  function evaluate(w: Vec): { hessian: number[][]; gradient: number[] } {
    const activations = matvec(design, w).map(sigmoid);
    const curvature = activations.map((y, n) => Math.max(pointWeights[n]! * y * (1 - y), CURVATURE_FLOOR));
    const weightedDesign = design.map((row, n) => row.map((v) => v * curvature[n]!));
    const hessian = symmetrise(matmul(transpose(weightedDesign), design));
    const gradient = matvec(
      transpose(design),
      activations.map((y, n) => pointWeights[n]! * (y - targets[n]!)),
    );
    return { hessian, gradient };
  }

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const { hessian, gradient } = evaluate(weights);
    const l = cholesky(hessian);
    const step = solveCholesky(l, gradient);
    const next = vecSub(weights, step);
    const delta = Math.sqrt(dot(vecSub(next, weights), vecSub(next, weights)));
    weights = next;
    if (delta <= tol) {
      return { weights, iterations: iteration + 1, converged: true };
    }
  }
  return { weights, iterations: maxIterations, converged: false };
}
