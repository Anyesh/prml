import { matmul, matAdd, matvec, pinv, solve, transpose, eye } from '../linalg/index.js';
import type { Mat, Vec } from '../types.js';

/**
 * Maps one input to a feature vector. The bias feature is the caller's responsibility to
 * request, not the basis's to assume, because the bias-variance demonstrations need a
 * basis that genuinely cannot represent a constant.
 */
export type BasisFunction = (x: number) => number[];

export interface BasisOptions {
  /** Prepends a constant 1 feature, the `phi_0` of PRML 3.2. */
  bias?: boolean;
}

/** `[1, x, x², …, x^degree]` when `bias` is set, dropping the leading 1 otherwise. */
export function polynomialBasis(degree: number, options?: BasisOptions): BasisFunction {
  const bias = options?.bias ?? true;
  const start = bias ? 0 : 1;
  return (x: number) => {
    const out: number[] = [];
    for (let p = start; p <= degree; p++) out.push(Math.pow(x, p));
    return out;
  };
}

/**
 * PRML 3.4. `scale` is the `s` in the exponent and controls how far each bump reaches,
 * which is the parameter the reader drags; it is not a standard deviation and the basis
 * is deliberately unnormalised, because the weights absorb any constant factor.
 */
export function gaussianBasis(centres: Vec, scale: number, options?: BasisOptions): BasisFunction {
  const bias = options?.bias ?? true;
  return (x: number) => {
    const bumps = centres.map((c) => Math.exp(-((x - c) ** 2) / (2 * scale * scale)));
    return bias ? [1, ...bumps] : bumps;
  };
}

/** PRML 3.5, the logistic sigmoid form. */
export function sigmoidalBasis(centres: Vec, scale: number, options?: BasisOptions): BasisFunction {
  const bias = options?.bias ?? true;
  return (x: number) => {
    const bumps = centres.map((c) => 1 / (1 + Math.exp(-(x - c) / scale)));
    return bias ? [1, ...bumps] : bumps;
  };
}

/** Row `n` is `phi(xs[n])`, the design matrix `Φ` of PRML 3.16. */
export function designMatrix(xs: Vec, phi: BasisFunction): number[][] {
  return xs.map((x) => phi(x));
}

/**
 * Maximum likelihood weights, PRML 3.15. Solved through the SVD-based pseudo-inverse of
 * `design` itself, never through the normal equations, because forming `ΦᵀΦ` squares the
 * matrix's condition number: on this chapter's own degree-9 fit, `cond(Φ)` is already 7e7,
 * which the normal equations would push past 5e15, into territory double precision cannot
 * resolve at all (PRML 3.1.1 makes the same point about the normal equations directly).
 */
export function maximumLikelihoodWeights(design: Mat, targets: Vec): number[] {
  return matvec(pinv(design), targets);
}

/**
 * Ridge solution, PRML 3.28: the normal equations with `lambda` added to the diagonal.
 * Unlike `maximumLikelihoodWeights`, forming `ΦᵀΦ` here is safe: `lambda` lifts every
 * eigenvalue of the normal matrix away from zero before it is squared, so the regularised
 * system never approaches the conditioning cliff the unregularised one falls off.
 */
export function regularisedWeights(design: Mat, targets: Vec, lambda: number): number[] {
  const phiT = transpose(design);
  const gram = matAdd(eye(phiT.length, lambda), matmul(phiT, design));
  const rhs = matvec(phiT, targets);
  return solve(gram, rhs);
}

/** Mean squared error between `Φw` and the targets. */
export function meanSquaredError(design: Mat, targets: Vec, weights: Vec): number {
  const predicted = matvec(design, weights);
  let sum = 0;
  for (let i = 0; i < targets.length; i++) sum += (predicted[i]! - targets[i]!) ** 2;
  return sum / targets.length;
}
