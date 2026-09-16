import { cholesky, inverse, matScale, norm, solveCholesky, symmetrise, vecAdd, vecSub } from '../linalg/index.js';
import { newtonRaphsonLogisticFit, type NewtonRaphsonOptions } from './irls.js';
import type { Mat, Vec } from '../types.js';

export interface LaplaceApproximation {
  readonly mode: number[];
  readonly covariance: number[][];
  readonly precision: number[][];
  readonly iterations: number;
  readonly converged: boolean;
}

/**
 * PRML 4.126-4.135, general form: Newton's method to the mode of an arbitrary log-density,
 * then a Gaussian fit with precision `A = -∇∇ ln f` evaluated there. `gradient` and
 * `hessian` are the caller's own analytic derivatives; for the skewed one-dimensional
 * posterior and the two-dimensional contour this section is built around, a numerically
 * differenced Hessian would be too noisy to trust in a figure meant to show exactly what
 * the Gaussian assumption throws away.
 *
 * A Newton step here solves `A · step = gradient(x)` and adds it, rather than the more
 * usual `x - Hessian⁻¹ gradient`: `hessian` is negative (semi-)definite at a maximum, so
 * `A = -hessian` is the positive-definite quantity Cholesky needs, and folding the sign
 * flip into the solve keeps that requirement local to this one function.
 */
export function laplaceApproximation(
  gradient: (x: Vec) => Vec,
  hessian: (x: Vec) => Mat,
  x0: Vec,
  options?: NewtonRaphsonOptions,
): LaplaceApproximation {
  const maxIterations = options?.maxIterations ?? 100;
  const tol = options?.tol ?? 1e-12;
  let x = [...x0];

  // Precision at the mode, PRML's `A = -∇∇ ln f(z0)`; recomputed after the loop exits so
  // it reflects the converged `x` rather than the iterate one step behind it.
  function precisionAt(point: Vec): number[][] {
    return symmetrise(matScale(hessian(point), -1));
  }

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const g = gradient(x);
    const precision = precisionAt(x);
    const l = cholesky(precision);
    const step = solveCholesky(l, g);
    const next = vecAdd(x, step);
    const delta = norm(vecSub(next, x));
    x = next;
    if (delta <= tol) {
      const finalPrecision = precisionAt(x);
      return { mode: x, covariance: symmetrise(inverse(finalPrecision)), precision: finalPrecision, iterations: iteration + 1, converged: true };
    }
  }
  const finalPrecision = precisionAt(x);
  return { mode: x, covariance: symmetrise(inverse(finalPrecision)), precision: finalPrecision, iterations: maxIterations, converged: false };
}

export interface GaussianPrior {
  readonly mean: Vec;
  readonly covariance: Mat;
}

export interface LaplaceLogisticPosterior {
  /** `wMAP`, the mean of the Gaussian approximation `q(w)`, PRML 4.144. */
  readonly mean: number[];
  /** `SN`, PRML 4.143-4.144. */
  readonly covariance: number[][];
  /** `SN⁻¹`. */
  readonly precision: number[][];
  readonly iterations: number;
  readonly converged: boolean;
}

/**
 * PRML 4.140-4.144: the Laplace approximation to the Bayesian logistic regression
 * posterior, specialised from `laplaceApproximation` above to reuse the IRLS Newton step
 * (`newtonRaphsonLogisticFit`) instead of a generic numerical Hessian, exactly as the book's
 * own narrative reaches for 4.3.3's algorithm again in section 4.5.1.
 */
export function fitLaplaceLogisticPosterior(
  design: Mat,
  targets: Vec,
  prior: GaussianPrior,
  options?: NewtonRaphsonOptions,
): LaplaceLogisticPosterior {
  const priorPrecision = inverse(prior.covariance);
  const result = newtonRaphsonLogisticFit(design, targets, prior.mean, priorPrecision, options);
  return {
    mean: result.weights,
    covariance: symmetrise(inverse(result.hessian)),
    precision: result.hessian,
    iterations: result.iterations,
    converged: result.converged,
  };
}
