import { dot, norm, vecSub } from '../linalg/index.js';
import type { Vec } from '../types.js';

/**
 * A kernel is symmetric in its two arguments (PRML 6.1): `k(x, x') = k(x', x)`. Every
 * constructor below upholds that by construction, never by a runtime check.
 */
export type KernelFunction = (x: Vec, xp: Vec) => number;

/** PRML 6.1 and 6.10: the kernel induced by an explicit feature map, `phi(x)^T phi(x')`. */
export function kernelFromFeatureMap(phi: (x: Vec) => Vec): KernelFunction {
  return (x, xp) => dot(phi(x), phi(xp));
}

/** The identity feature map, `phi(x) = x`, so `k(x, x') = x^T x'`. */
export function linearKernel(): KernelFunction {
  return (x, xp) => dot(x, xp);
}

/**
 * PRML 6.11-6.12 generalised to `(x^T x' + c)^degree`. `c = 0` is the homogeneous kernel of
 * (6.11), containing only degree-`degree` monomials; `c > 0` mixes in every lower degree too,
 * as (6.12) shows by direct expansion for `degree = 2`. `c` is required because the two are
 * different feature spaces and a default would silently pick one of them.
 */
export function polynomialKernel(degree: number, c: number): KernelFunction {
  return (x, xp) => Math.pow(dot(x, xp) + c, degree);
}

function squaredDistance(x: Vec, xp: Vec): number {
  let acc = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i]! - xp[i]!;
    acc += d * d;
  }
  return acc;
}

/**
 * PRML 6.23 when `lengthScale` is a single number, and 6.71/6.72 (automatic relevance
 * determination) when it is one value per input dimension: `exp(-0.5 * sum_i (xi-xi')^2 /
 * lengthScale_i^2)`. The two equations are the same construction, an isotropic scale being
 * the special case where every axis shares one.
 */
export function rbfKernel(lengthScale: number | Vec): KernelFunction {
  return (x, xp) => {
    let acc = 0;
    for (let i = 0; i < x.length; i++) {
      const l = typeof lengthScale === 'number' ? lengthScale : lengthScale[i]!;
      const d = (x[i]! - xp[i]!) / l;
      acc += d * d;
    }
    return Math.exp(-0.5 * acc);
  };
}

/**
 * The same Gaussian kernel as `rbfKernel`, in the parameterisation the support vector
 * literature uses: `gamma = 1 / (2 * lengthScale^2)`. Both exist because converting at every
 * call site is how a factor of two ends up in the wrong place.
 */
export function rbfKernelGamma(gamma: number): KernelFunction {
  return (x, xp) => Math.exp(-gamma * squaredDistance(x, xp));
}

/** PRML 6.56: the Ornstein-Uhlenbeck / exponential kernel, `exp(-theta * ||x - x'||)`. */
export function exponentialKernel(theta: number): KernelFunction {
  return (x, xp) => Math.exp(-theta * norm(vecSub(x, xp)));
}

export interface CompositeKernelParams {
  readonly theta0: number;
  readonly theta1: number;
  readonly theta2: number;
  readonly theta3: number;
}

/**
 * PRML 6.63: a squared-exponential term plus a constant plus a linear term, each switched
 * independently by its own theta. `theta3` alone recovers a parametric linear model inside
 * the kernel, which is the point PRML makes directly under this equation.
 */
export function compositeKernel(params: CompositeKernelParams): KernelFunction {
  const { theta0, theta1, theta2, theta3 } = params;
  return (x, xp) => theta0 * Math.exp((-theta1 / 2) * squaredDistance(x, xp)) + theta2 + theta3 * dot(x, xp);
}

/**
 * Partial derivatives of (6.63) with respect to theta0..theta3, in that order, each shaped
 * as a `KernelFunction` so `gramMatrix` builds `dCN/dtheta_i` from one exactly the way it
 * builds `CN` from `compositeKernel` itself (needed by `gpLogMarginalLikelihoodGradient`).
 */
export function compositeKernelPartials(params: CompositeKernelParams): readonly KernelFunction[] {
  const { theta0, theta1 } = params;
  const gaussianTerm: KernelFunction = (x, xp) => Math.exp((-theta1 / 2) * squaredDistance(x, xp));
  return [
    gaussianTerm,
    (x, xp) => -0.5 * theta0 * squaredDistance(x, xp) * gaussianTerm(x, xp),
    () => 1,
    (x, xp) => dot(x, xp),
  ];
}

/**
 * PRML 6.37. Kept despite the book's own warning that its Gram matrix is "in general not
 * positive semidefinite", because that failure is exactly what 6.2's construction rules are
 * for: a kernel worth naming as a counterexample, not a hypothetical one.
 */
export function sigmoidKernel(a: number, b: number): KernelFunction {
  return (x, xp) => Math.tanh(a * dot(x, xp) + b);
}
