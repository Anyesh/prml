import type { Mat, Vec } from '../types.js';
import { sigmoid } from '../numeric.js';
import { dot, inverse, logDet, matAdd, matScale, matvec, outer, solve, symmetrise, vecAdd, vecScale } from '../linalg/index.js';

const LAMBDA_SINGULARITY_GUARD = 1e-8;
const LAMBDA_AT_ZERO = 0.125;

/** PRML 10.141: `lambda(xi) = [sigma(xi) - 1/2] / (2 xi)`, extended to `1/8` at `xi = 0` by its own limit. */
export function logisticLambda(xi: number): number {
  if (Math.abs(xi) < LAMBDA_SINGULARITY_GUARD) return LAMBDA_AT_ZERO;
  return (sigmoid(xi) - 0.5) / (2 * xi);
}

/** PRML 10.144: touches `sigma(x)` exactly at `x = xi` and `x = -xi`, and lies below it everywhere else. */
export function logisticLocalBound(x: number, xi: number): number {
  return sigmoid(xi) * Math.exp((x - xi) / 2 - logisticLambda(xi) * (x * x - xi * xi));
}

export interface VariationalLogisticPrior {
  readonly mean: Vec;
  readonly cov: Mat;
}

export interface VariationalLogisticPosterior {
  readonly mean: Vec;
  readonly cov: Mat;
}

/**
 * `q(w)` in closed form for a fixed variational parameter per observation, PRML
 * 10.157-10.159: the local bound turns the logistic likelihood into something quadratic in
 * `w`, so this is Bayesian linear regression's precision-and-mean update with `2 lambda(xi_n)`
 * standing in for a noise precision that varies by observation.
 */
export function variationalLogisticUpdate(design: Mat, targets: Vec, prior: VariationalLogisticPrior, xi: Vec): VariationalLogisticPosterior {
  const priorPrecision = inverse(prior.cov);

  let precision = priorPrecision;
  let rhs = matvec(priorPrecision, prior.mean);
  for (let n = 0; n < design.length; n++) {
    const phi = design[n]!;
    const l = logisticLambda(xi[n]!);
    precision = matAdd(precision, matScale(outer(phi, phi), 2 * l));
    rhs = vecAdd(rhs, vecScale(phi, targets[n]! - 0.5));
  }
  precision = symmetrise(precision);
  const cov = symmetrise(inverse(precision));
  const mean = matvec(cov, rhs);
  return { mean, cov };
}

/** PRML 10.163-10.164: `xi_n^2 = phi_n^T (S + m m^T) phi_n`, the second moment of `phi_n^T w` under `q(w)`. */
export function updateXi(design: Mat, mean: Vec, cov: Mat): number[] {
  return design.map((phi) => {
    const projected = matvec(cov, phi);
    const quad = dot(phi, projected);
    const linear = dot(phi, mean);
    return Math.sqrt(Math.max(quad + linear * linear, 0));
  });
}

/**
 * PRML 10.161: the Gaussian KL terms between `q(w)` and the prior, plus a per-observation
 * term that is exactly what remains of the local bound once `w` has been integrated out.
 */
export function variationalLogisticLowerBound(
  prior: VariationalLogisticPrior,
  posterior: VariationalLogisticPosterior,
  xi: Vec,
): number {
  const priorPrecision = inverse(prior.cov);
  const quadPosterior = dot(posterior.mean, solve(posterior.cov, posterior.mean));
  const quadPrior = dot(prior.mean, matvec(priorPrecision, prior.mean));

  let total = 0.5 * (logDet(symmetrise(posterior.cov)) - logDet(symmetrise(prior.cov))) + 0.5 * quadPosterior - 0.5 * quadPrior;
  for (const xiN of xi) {
    total += Math.log(sigmoid(xiN)) - xiN / 2 + logisticLambda(xiN) * xiN * xiN;
  }
  return total;
}

export interface VariationalLogisticFitResult {
  readonly posteriorHistory: readonly VariationalLogisticPosterior[];
  readonly xiHistory: readonly Vec[];
  readonly lowerBoundHistory: readonly number[];
}

/** Alternates `variationalLogisticUpdate` and `updateXi` for `rounds` sweeps, from a fixed `xiInit`. */
export function variationalLogisticFit(
  design: Mat,
  targets: Vec,
  prior: VariationalLogisticPrior,
  xiInit: Vec,
  rounds: number,
): VariationalLogisticFitResult {
  const posteriorHistory: VariationalLogisticPosterior[] = [];
  const xiHistory: Vec[] = [];
  const lowerBoundHistory: number[] = [];

  let xi = xiInit;
  for (let r = 0; r < rounds; r++) {
    const posterior = variationalLogisticUpdate(design, targets, prior, xi);
    xi = updateXi(design, posterior.mean, posterior.cov);
    posteriorHistory.push(posterior);
    xiHistory.push(xi);
    lowerBoundHistory.push(variationalLogisticLowerBound(prior, posterior, xi));
  }
  return { posteriorHistory, xiHistory, lowerBoundHistory };
}

/**
 * PRML 10.6.3: an evidence-style re-estimation of an isotropic prior precision `alpha`,
 * given the converged `q(w)`. Mirrors the fixed point ch3's evidence approximation reaches
 * for a point estimate of `alpha`, now applied to the variational posterior instead of a
 * MAP weight vector.
 */
export function updateLogisticAlpha(posterior: VariationalLogisticPosterior, alpha: number): number {
  const m = posterior.mean.length;
  let traceTerm = 0;
  for (let i = 0; i < m; i++) traceTerm += posterior.cov[i]![i]!;
  const gamma = m - alpha * traceTerm;
  const wtw = dot(posterior.mean, posterior.mean);
  return gamma / wtw;
}
