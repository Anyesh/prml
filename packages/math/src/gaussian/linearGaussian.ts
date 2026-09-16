import { inverse } from '../linalg/decompose.js';
import { matAdd, matmul, matvec, transpose, vecAdd, vecSub } from '../linalg/core.js';
import type { Mat, Vec } from '../types.js';
import type { MvnParams } from '../distributions/mvn.js';

/** A Gaussian specified by its precision rather than its covariance, PRML's `N(x | mu, Lambda^-1)`. */
export interface GaussianPrecisionParams {
  readonly mean: Vec;
  readonly precision: Mat;
}

/**
 * The linear-Gaussian likelihood `p(y | x) = N(y | Ax + b, L^-1)` of PRML 2.114, the
 * building block behind every "Gaussian prior, linear-Gaussian observation" model in the
 * book, Bayesian linear regression's predictive distribution (3.58-3.59) included.
 */
export interface LinearGaussianLikelihood {
  readonly a: Mat;
  readonly b: Vec;
  readonly precision: Mat;
}

/**
 * Marginal `p(y) = N(y | A mu + b, L^-1 + A Lambda^-1 A^T)`, PRML 2.114-2.115.
 *
 * Integrating out `x` adds the likelihood's own noise covariance to `x`'s uncertainty
 * pushed through `A`, which is why this covariance is never smaller than either term
 * alone.
 */
export function linearGaussianMarginal(
  prior: GaussianPrecisionParams,
  likelihood: LinearGaussianLikelihood,
): MvnParams {
  const { a, b, precision: l } = likelihood;
  const lambdaInv = inverse(prior.precision);
  const lInv = inverse(l);
  const mean = vecAdd(matvec(a, prior.mean), b);
  const cov = matAdd(lInv, matmul(matmul(a, lambdaInv), transpose(a)));
  return { mean, cov };
}

/**
 * Posterior `p(x | y)`, PRML 2.116-2.117: covariance `Sigma = (Lambda + A^T L A)^-1`,
 * mean `Sigma {A^T L (y - b) + Lambda mu}`.
 *
 * This is Bayes' theorem for Gaussian variables done directly on the precision-form
 * prior and the linear-Gaussian likelihood, without ever building the joint distribution
 * over `(x, y)` that `mvnConditional` would need.
 */
export function linearGaussianPosterior(
  prior: GaussianPrecisionParams,
  likelihood: LinearGaussianLikelihood,
  y: Vec,
): MvnParams {
  const { a, b, precision: l } = likelihood;
  const aT = transpose(a);
  const cov = inverse(matAdd(prior.precision, matmul(matmul(aT, l), a)));
  const residual = vecSub(y, b);
  const rhs = vecAdd(matvec(matmul(aT, l), residual), matvec(prior.precision, prior.mean));
  const mean = matvec(cov, rhs);
  return { mean, cov };
}
