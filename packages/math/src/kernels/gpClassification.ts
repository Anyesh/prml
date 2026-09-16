import {
  cholesky,
  dot,
  inverse,
  jitter as addJitter,
  matvec,
  solveCholesky,
  symmetrise,
  zeros,
} from '../linalg/index.js';
import { laplaceApproximation } from '../classification/laplace.js';
import { logisticGaussianConvolution } from '../classification/probit.js';
import { sigmoid } from '../numeric.js';
import type { Mat, Vec } from '../types.js';
import { gramMatrix, kernelVector } from './combine.js';
import type { KernelFunction } from './functions.js';

const DEFAULT_NU = 1e-6;

export interface GPClassificationModel {
  readonly kernel: KernelFunction;
  readonly trainX: readonly Vec[];
  readonly trainT: Vec;
  readonly nu: number;
  /** `CN` of PRML 6.75, the Gram matrix plus the numerical noise term `nu`. */
  readonly covariance: Mat;
  /** `a*`, the mode of `Psi(aN)`, PRML 6.84. */
  readonly mode: number[];
  /** `tN - sigma(a*)`, reused directly by the predictive mean (6.87). */
  readonly residual: number[];
  /** `sigma(a*_n)(1 - sigma(a*_n))`, the diagonal of `WN` at the mode. */
  readonly w: number[];
  readonly iterations: number;
  readonly converged: boolean;
}

/**
 * PRML 6.77-6.86: the Laplace approximation to the Gaussian process classification
 * posterior over `aN`. Reuses `laplaceApproximation` (chapter 4's generic Newton solver to
 * the mode of an arbitrary log-density) rather than re-deriving Newton's method here: the
 * gradient (6.81) and Hessian (6.82) of `Psi(aN)` are exactly the `gradient`/`hessian`
 * callbacks that solver already takes, and the fixed point it converges to is (6.84) by
 * construction, whichever way the intermediate algebra is arranged. The book's own (6.83)
 * rearranges the same Newton step to avoid inverting `CN` explicitly; forming `CN^-1` once
 * here instead is the cost of sharing one Newton implementation across chapters 4 and 6,
 * and is affordable at the training-set sizes this chapter's widgets use.
 */
export function fitGPClassificationLaplace(
  kernel: KernelFunction,
  trainX: readonly Vec[],
  trainT: Vec,
  nu = DEFAULT_NU,
): GPClassificationModel {
  const n = trainX.length;
  const covariance = symmetrise(addJitter(gramMatrix(kernel, trainX), nu));
  const covInv = inverse(covariance);

  function gradient(a: Vec): number[] {
    const covInvA = matvec(covInv, a);
    return a.map((_, i) => trainT[i]! - sigmoid(a[i]!) - covInvA[i]!);
  }
  function hessian(a: Vec): Mat {
    return covInv.map((row, i) => {
      const s = sigmoid(a[i]!);
      const wi = s * (1 - s);
      return row.map((v, j) => -(v + (i === j ? wi : 0)));
    });
  }

  const result = laplaceApproximation(gradient, hessian, zeros(n));
  const mode = result.mode;
  const sigmaStar = mode.map(sigmoid);
  const residual = trainT.map((t, i) => t - sigmaStar[i]!);
  const w = sigmaStar.map((s) => s * (1 - s));
  return { kernel, trainX, trainT, nu, covariance, mode, residual, w, iterations: result.iterations, converged: result.converged };
}

export interface GPClassificationPrediction {
  /** `E[a_{N+1} | tN]`, PRML 6.87. */
  readonly meanA: number;
  /** `var[a_{N+1} | tN]`, PRML 6.88. */
  readonly varianceA: number;
  /** `p(t_{N+1} = 1 | tN)`, approximating the intractable (6.76) via the (4.153) convolution. */
  readonly probability: number;
}

export function gpClassificationPredict(model: GPClassificationModel, x: Vec): GPClassificationPrediction {
  const k = kernelVector(model.kernel, model.trainX, x);
  const meanA = dot(k, model.residual);

  const c = model.kernel(x, x) + model.nu;
  const wInvPlusCN = model.covariance.map((row, i) => row.map((v, j) => v + (i === j ? 1 / model.w[i]! : 0)));
  const l = cholesky(symmetrise(wInvPlusCN));
  const solved = solveCholesky(l, k);
  // Clamped at zero: floating-point cancellation in c - k^T(...)^-1 k can push a variance
  // that is mathematically non-negative fractionally below it.
  const varianceA = Math.max(0, c - dot(k, solved));
  return { meanA, varianceA, probability: logisticGaussianConvolution(meanA, varianceA) };
}
