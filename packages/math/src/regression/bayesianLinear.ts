import {
  cholesky,
  eye,
  matAdd,
  matScale,
  matmul,
  matvec,
  outer,
  quadForm,
  solveCholesky,
  symmetrise,
  transpose,
  vecAdd,
  vecScale,
  zeros,
} from '../linalg/index.js';
import { standardNormal } from '../rng.js';
import type { Mat, Rng, Vec } from '../types.js';

export interface Hyperparameters {
  /** Prior precision over the weights, the `alpha` of PRML 3.52. */
  readonly alpha: number;
  /** Noise precision, the `beta` of PRML 3.8. */
  readonly beta: number;
}

export interface WeightPosterior {
  /** `m_N`, PRML 3.53. */
  readonly mean: number[];
  /** `S_N`, PRML 3.54. */
  readonly cov: number[][];
  /** `S_N⁻¹`. Carried rather than recomputed because the sequential update adds to it directly. */
  readonly precision: number[][];
}

/** Inverts an SPD matrix from its own Cholesky factor, column by column. */
function invertFromFactor(l: Mat): number[][] {
  const n = l.length;
  const cols = Array.from({ length: n }, (_, j) => {
    const e = zeros(n);
    e[j] = 1;
    return solveCholesky(l, e);
  });
  return symmetrise(transpose(cols));
}

/**
 * PRML 3.53-3.54, for the isotropic prior `N(0, alpha⁻¹ I)`.
 *
 * `cov` is formed explicitly, which `linalg`'s own documentation warns against for solving
 * systems. It is warranted here and only here: the widget draws the posterior as a
 * covariance ellipse, so the matrix is the displayed object rather than an intermediate.
 */
export function weightPosterior(design: Mat, targets: Vec, hyper: Hyperparameters): WeightPosterior {
  const { alpha, beta } = hyper;
  const phiT = transpose(design);
  const dimension = phiT.length;
  const precision = symmetrise(matAdd(eye(dimension, alpha), matScale(matmul(phiT, design), beta)));
  const l = cholesky(precision);
  const cov = invertFromFactor(l);
  const mean = vecScale(matvec(cov, matvec(phiT, targets)), beta);
  return { mean, cov, precision };
}

/**
 * Folds one more observation into an existing posterior, PRML 3.50-3.51.
 *
 * This must be a rank-one update rather than a refit over all data seen so far. The point
 * of the sequential learning figure is that the posterior after n points is the prior for
 * point n+1, and a refit would make that claim while quietly doing something else.
 */
export function updatePosterior(
  prior: WeightPosterior,
  phiX: Vec,
  target: number,
  beta: number,
): WeightPosterior {
  const precision = symmetrise(matAdd(prior.precision, matScale(outer(phiX, phiX), beta)));
  const rhs = vecAdd(matvec(prior.precision, prior.mean), vecScale(phiX, beta * target));
  const l = cholesky(precision);
  const mean = solveCholesky(l, rhs);
  const cov = invertFromFactor(l);
  return { mean, cov, precision };
}

/** The prior `N(0, alpha⁻¹I)` in the same shape as a posterior, so the sequential update can start from it. */
export function isotropicPrior(dimension: number, alpha: number): WeightPosterior {
  return {
    mean: zeros(dimension),
    cov: eye(dimension, 1 / alpha),
    precision: eye(dimension, alpha),
  };
}

export interface Predictive {
  readonly mean: number;
  /** PRML 3.59. Includes the noise term, so it never falls below `1/beta` however much data arrives. */
  readonly variance: number;
}

/** PRML 3.58-3.59. */
export function predictive(phiX: Vec, posterior: WeightPosterior, beta: number): Predictive {
  const mean = phiX.reduce((s, v, i) => s + v * posterior.mean[i]!, 0);
  const variance = 1 / beta + quadForm(phiX, posterior.cov, phiX);
  return { mean, variance };
}

/**
 * Draws `count` weight vectors from the posterior, which the widget renders as sampled
 * regression lines. Sampling the weights rather than the function values is what makes
 * the drawn lines coherent across x rather than independently wobbling at each point.
 */
export function sampleWeights(rng: Rng, posterior: WeightPosterior, count: number): number[][] {
  const l = cholesky(posterior.cov);
  const dimension = posterior.mean.length;
  const draws: number[][] = [];
  for (let s = 0; s < count; s++) {
    const z = Array.from({ length: dimension }, () => standardNormal(rng));
    const lz = matvec(l, z);
    draws.push(posterior.mean.map((m, i) => m + lz[i]!));
  }
  return draws;
}

/**
 * The equivalent kernel `k(x, x')` of PRML 3.62, as a function of two design rows.
 *
 * Smoothing weights, not a Mercer kernel: they sum to one over the training inputs and
 * may be negative, which is the property the figure exists to show.
 */
export function equivalentKernel(
  posterior: WeightPosterior,
  beta: number,
): (phiA: Vec, phiB: Vec) => number {
  return (phiA: Vec, phiB: Vec) => beta * quadForm(phiA, posterior.cov, phiB);
}
