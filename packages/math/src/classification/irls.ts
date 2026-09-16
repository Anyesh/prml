import {
  cholesky,
  dot,
  matAdd,
  matZeros,
  matmul,
  matvec,
  norm,
  solveCholesky,
  symmetrise,
  transpose,
  vecAdd,
  vecSub,
  zeros,
} from '../linalg/index.js';
import { sigmoid } from '../numeric.js';
import type { Mat, Vec } from '../types.js';

/** `σ(wᵀφ)`, PRML 4.87 with the sigmoid link. */
export function logisticPredict(weights: Vec, features: Vec): number {
  return sigmoid(dot(weights, features));
}

function softplus(a: number): number {
  return Math.max(a, 0) + Math.log1p(Math.exp(-Math.abs(a)));
}

/**
 * PRML 4.90, the cross-entropy error, computed from the activations directly rather than
 * through `ln(sigmoid(a))`. On separable data the widget for this section drives `w` (and
 * so `a`) toward infinity on purpose, at which point a naive `-ln y` evaluates `ln(0)`; the
 * softplus identity `E = softplus(a) - t*a` stays finite for every finite `a` and is exactly
 * equal to the naive form wherever that form is itself finite.
 */
export function crossEntropyError(design: Mat, targets: Vec, weights: Vec): number {
  let sum = 0;
  for (let n = 0; n < design.length; n++) {
    const a = dot(weights, design[n]!);
    sum += softplus(a) - targets[n]! * a;
  }
  return sum;
}

/** PRML 4.91: the gradient collapses to `Φᵀ(y - t)` once the sigmoid's own derivative cancels. */
export function crossEntropyGradient(design: Mat, targets: Vec, weights: Vec): number[] {
  const activations = matvec(design, weights).map(sigmoid);
  const residual = activations.map((y, n) => y - targets[n]!);
  return matvec(transpose(design), residual);
}

/**
 * `y(1-y)` underflows to exactly 0 once the sigmoid saturates past roughly 1e-8 from 0 or 1.
 * A zero here would zero out an entire diagonal entry of `R` and, on data with few enough
 * points relative to features, make `ΦᵀRΦ` singular right at the moment separable data is
 * pushing weights toward infinity - the one place this chapter deliberately drives IRLS. The
 * floor keeps the linear solve well-posed at every iteration without capping how large the
 * weights themselves are allowed to grow.
 */
const R_FLOOR = 1e-10;

interface NewtonRaphsonResult {
  readonly weights: number[];
  readonly hessian: number[][];
  readonly iterations: number;
  readonly converged: boolean;
  readonly history: readonly { readonly iteration: number; readonly weights: number[] }[];
}

export interface NewtonRaphsonOptions {
  readonly initialWeights?: Vec;
  readonly maxIterations?: number;
  readonly tol?: number;
}

/**
 * The Newton-Raphson step shared by plain IRLS (PRML 4.92-4.100) and, with a nonzero prior
 * precision, the MAP estimate of Bayesian logistic regression (PRML 4.142-4.143): the
 * S0⁻¹ term of 4.143 is exactly `priorPrecision`, and it vanishes to recover 4.97 when the
 * prior is flat. Keeping one implementation means a fix to the weighting-matrix floor above
 * applies to both call sites at once, instead of two copies drifting apart.
 */
export function newtonRaphsonLogisticFit(
  design: Mat,
  targets: Vec,
  priorMean: Vec,
  priorPrecision: Mat,
  options?: NewtonRaphsonOptions,
): NewtonRaphsonResult {
  const dimension = design[0]?.length ?? 0;
  const maxIterations = options?.maxIterations ?? 50;
  const tol = options?.tol ?? 1e-10;
  let weights = options?.initialWeights ? [...options.initialWeights] : zeros(dimension);
  const history: { iteration: number; weights: number[] }[] = [];

  // Both the Hessian and gradient of the log posterior at the given weights, PRML
  // 4.96-4.98 (likelihood term) plus the S0^-1 term of 4.143 (prior term).
  function evaluate(w: Vec): { hessian: number[][]; gradient: number[] } {
    const activations = matvec(design, w).map(sigmoid);
    const rDiag = activations.map((y) => Math.max(y * (1 - y), R_FLOOR));
    const weightedDesign = design.map((row, n) => row.map((v) => v * rDiag[n]!));
    const hessian = symmetrise(matAdd(priorPrecision, matmul(transpose(weightedDesign), design)));
    const gradLikelihood = matvec(
      transpose(design),
      activations.map((y, n) => y - targets[n]!),
    );
    const gradPrior = matvec(priorPrecision, vecSub(w, priorMean));
    return { hessian, gradient: vecAdd(gradPrior, gradLikelihood) };
  }

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    const { hessian, gradient } = evaluate(weights);
    const l = cholesky(hessian);
    const step = solveCholesky(l, gradient);
    const next = vecSub(weights, step);
    history.push({ iteration, weights: next });

    const delta = norm(vecSub(next, weights));
    weights = next;
    if (delta <= tol) {
      // Re-evaluated at the converged weights, so the reported curvature is the Hessian
      // at the mode (PRML 4.143) rather than at the penultimate iterate.
      return { weights, hessian: evaluate(weights).hessian, iterations: iteration + 1, converged: true, history };
    }
  }
  return { weights, hessian: evaluate(weights).hessian, iterations: maxIterations, converged: false, history };
}

export interface IrlsFit {
  readonly weights: number[];
  /** `ΦᵀRΦ` at the converged weights, PRML 4.97. */
  readonly precision: number[][];
  readonly iterations: number;
  readonly converged: boolean;
  readonly history: readonly { readonly iteration: number; readonly weights: number[] }[];
}

/** PRML 4.99-4.100: Newton-Raphson on the cross-entropy error, with a flat (zero-precision) prior. */
export function irlsFit(design: Mat, targets: Vec, options?: NewtonRaphsonOptions): IrlsFit {
  const dimension = design[0]?.length ?? 0;
  const flatPrior = matZeros(dimension, dimension);
  const result = newtonRaphsonLogisticFit(design, targets, zeros(dimension), flatPrior, options);
  return {
    weights: result.weights,
    precision: result.hessian,
    iterations: result.iterations,
    converged: result.converged,
    history: result.history,
  };
}
