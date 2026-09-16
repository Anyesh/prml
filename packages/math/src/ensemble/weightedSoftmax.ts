import { cholesky, dot, matZeros, solveCholesky, symmetrise } from '../linalg/index.js';
import { softmax } from '../numeric.js';
import type { Mat, Vec } from '../types.js';

/**
 * Same role as `weightedLogistic.ts`'s CURVATURE_FLOOR, but the K-class softmax's Hessian
 * cannot be floored entry-by-entry the way the binary case's diagonal `R` can, because a
 * single point's curvature is spread across every (k, l) block at once. Added to the
 * diagonal of the assembled reduced Hessian at every iteration, unconditionally: once a
 * gate has learned a confident, sharp partition, some responsibilities sit at exactly 0
 * or 1 for every point, which drives that Hessian toward singular right where the fit is
 * most confident. The floor keeps the linear solve well-posed without capping how sharp
 * the fitted partition is allowed to become.
 */
const RIDGE_FLOOR = 1e-10;

export interface WeightedSoftmaxFit {
  /** K x D. Row 0 is always the zero vector; see the module doc for why. */
  readonly weights: Mat;
  readonly iterations: number;
  readonly converged: boolean;
}

export interface WeightedSoftmaxFitOptions {
  readonly maxIterations?: number;
  readonly tol?: number;
}

function predict(design: Mat, weights: Mat): number[][] {
  return design.map((phi) => softmax(weights.map((w) => dot(w, phi))));
}

/**
 * Newton-Raphson on the per-point-weighted multiclass cross-entropy error
 * `sum_n pointWeights[n] * -sum_k targets[n][k] * ln y_nk`, `y = softmax(W phi_n)`, the
 * K-class generalisation of `weightedLogistic.ts`'s binary IRLS (PRML 4.104-4.109 with a
 * per-point weight). `targets[n]` may be any distribution over classes summing to 1, not
 * only one-hot: the gradient identity `y_nk - t_nk` and the Hessian identity
 * `y_nk(I_kl - y_nl)` below only ever use `targets[n][k]` linearly, so nothing in the
 * derivation assumed a hard label. `pointWeights = 1` with one-hot `targets` recovers
 * plain multiclass logistic regression; the mixture-of-experts gate (14.5.3) instead
 * passes uniform `pointWeights` with the EM responsibilities as soft targets, the direct
 * K-class generalisation of 4.106's cross-entropy the same way `weightedLogisticFit`
 * already generalises the K=2 case.
 *
 * The K-class softmax is invariant to adding any constant vector to every row of `W`
 * (`softmax(a) = softmax(a + c)`), so the naive K*D-parameter Hessian is exactly
 * singular, not merely ill-conditioned, at every point, not only at convergence. This is
 * resolved structurally rather than papered over with a ridge: class 0's weights are
 * fixed at the zero vector (a reference class) and only the remaining `(K-1)*D`
 * parameters are solved for, coupled through the off-diagonal Hessian blocks
 * `sum_n pointWeights[n] * y_nk * y_nl * phi_n phi_n^T` (k != l) that a softmax's shared
 * normaliser induces; dropping those blocks would not be Newton's method on this
 * objective. `weights[0]` in the result is always the zero vector by this convention.
 */
export function weightedSoftmaxFit(
  design: Mat,
  targets: Mat,
  pointWeights: Vec,
  options?: WeightedSoftmaxFitOptions,
): WeightedSoftmaxFit {
  const dimension = design[0]?.length ?? 0;
  const numClasses = targets[0]?.length ?? 0;
  const freeClasses = numClasses - 1;
  const paramDim = freeClasses * dimension;
  const maxIterations = options?.maxIterations ?? 100;
  const tol = options?.tol ?? 1e-12;

  // Class 0's row stays the zero vector throughout; only rows 1..numClasses-1 are fit.
  let freeWeights: number[][] = Array.from({ length: freeClasses }, () => new Array(dimension).fill(0));

  function fullWeights(free: Mat): Mat {
    return [new Array(dimension).fill(0), ...free];
  }

  function evaluate(free: Mat): { hessian: number[][]; gradient: number[] } {
    const weights = fullWeights(free);
    const predictions = predict(design, weights);
    const gradient = new Array(paramDim).fill(0);
    const hessian = matZeros(paramDim, paramDim);

    design.forEach((phi, n) => {
      const y = predictions[n]!;
      const t = targets[n]!;
      const w = pointWeights[n]!;

      for (let k = 1; k < numClasses; k++) {
        const fk = k - 1;
        const gk = w * (y[k]! - t[k]!);
        for (let d = 0; d < dimension; d++) gradient[fk * dimension + d]! += gk * phi[d]!;
      }

      for (let k = 1; k < numClasses; k++) {
        const fk = k - 1;
        for (let l = 1; l < numClasses; l++) {
          const fl = l - 1;
          const indicator = k === l ? 1 : 0;
          const curvature = w * y[k]! * (indicator - y[l]!);
          if (curvature === 0) continue;
          for (let d1 = 0; d1 < dimension; d1++) {
            const row = hessian[fk * dimension + d1]!;
            const c = curvature * phi[d1]!;
            for (let d2 = 0; d2 < dimension; d2++) {
              row[fl * dimension + d2]! += c * phi[d2]!;
            }
          }
        }
      }
    });

    // Block (k, l) and block (l, k) are mathematically transposes of each other
    // (`y_k * y_l` computed in a different multiplication order per block), which
    // floating point does not guarantee bit-identical, and the Cholesky routine below
    // requires exact symmetry rather than tolerating that rounding.
    const symmetric = symmetrise(hessian);
    for (let i = 0; i < paramDim; i++) symmetric[i]![i]! += RIDGE_FLOOR;
    return { hessian: symmetric, gradient };
  }

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const { hessian, gradient } = evaluate(freeWeights);
    const l = cholesky(hessian);
    const step = solveCholesky(l, gradient);

    let delta = 0;
    const next: number[][] = freeWeights.map((row, fk) =>
      row.map((v, d) => {
        const updated = v - step[fk * dimension + d]!;
        delta += (updated - v) * (updated - v);
        return updated;
      }),
    );
    freeWeights = next;
    if (Math.sqrt(delta) <= tol) {
      return { weights: fullWeights(freeWeights), iterations: iteration + 1, converged: true };
    }
  }
  return { weights: fullWeights(freeWeights), iterations: maxIterations, converged: false };
}
