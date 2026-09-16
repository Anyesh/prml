import { inverse, matAdd, matScale, matZeros, matvec, quadForm, vecSub } from '../linalg/index.js';
import { mvnLogPdf } from '../distributions/index.js';
import { softmax } from '../numeric.js';
import { columnMeans, scatterMatrix } from './stats.js';
import type { Mat, Vec } from '../types.js';

export interface SharedCovarianceFit {
  readonly priors: number[];
  readonly means: number[][];
  readonly covariance: number[][];
}

/**
 * Maximum-likelihood fit of PRML 4.73-4.76, generalised from two classes to K with a single
 * pooled covariance shared across every class (the paragraph immediately after 4.76).
 */
export function fitSharedCovarianceGaussian(classPoints: readonly Mat[]): SharedCovarianceFit {
  const total = classPoints.reduce((sum, points) => sum + points.length, 0);
  const means = classPoints.map(columnMeans);
  const priors = classPoints.map((points) => points.length / total);
  const dimension = means[0]?.length ?? 0;
  let pooled = matZeros(dimension, dimension);
  classPoints.forEach((points, k) => {
    pooled = matAdd(pooled, scatterMatrix(points, means[k]!));
  });
  return { priors, means, covariance: matScale(pooled, 1 / total) };
}

export interface SeparateCovarianceFit {
  readonly priors: number[];
  readonly means: number[][];
  readonly covariances: number[][][];
}

/**
 * The quadratic-discriminant extension the book describes only in prose, right after 4.67
 * ("cancellations will no longer occur, and we will obtain quadratic functions of x"): each
 * class keeps its own maximum-likelihood covariance instead of sharing one.
 */
export function fitSeparateCovarianceGaussian(classPoints: readonly Mat[]): SeparateCovarianceFit {
  const total = classPoints.reduce((sum, points) => sum + points.length, 0);
  const means = classPoints.map(columnMeans);
  const priors = classPoints.map((points) => points.length / total);
  const covariances = classPoints.map((points, k) => matScale(scatterMatrix(points, means[k]!), 1 / points.length));
  return { priors, means, covariances };
}

/** PRML 4.63 activations from Gaussian class-conditionals, softmaxed by 4.62. */
export function posteriorSharedCovariance(x: Vec, fit: SharedCovarianceFit): number[] {
  const activations = fit.means.map(
    (mean, k) => mvnLogPdf(x, { mean, cov: fit.covariance }) + Math.log(fit.priors[k]!),
  );
  return softmax(activations);
}

/** As `posteriorSharedCovariance`, but each class reads its own covariance from `fit`. */
export function posteriorSeparateCovariance(x: Vec, fit: SeparateCovarianceFit): number[] {
  const activations = fit.means.map(
    (mean, k) => mvnLogPdf(x, { mean, cov: fit.covariances[k]! }) + Math.log(fit.priors[k]!),
  );
  return softmax(activations);
}

export interface LinearBoundary {
  readonly w: number[];
  readonly w0: number;
}

/**
 * PRML 4.66-4.67: the closed-form two-class shared-covariance boundary. Equivalent to
 * calling `posteriorSharedCovariance` and thresholding at 0.5, but this exposes the weight
 * vector directly, which is what the boundary-as-a-line widget draws.
 */
export function sharedCovarianceLinearBoundary(fit: SharedCovarianceFit): LinearBoundary {
  const [mean1, mean2] = fit.means;
  if (fit.means.length !== 2 || mean1 === undefined || mean2 === undefined) {
    throw new Error('sharedCovarianceLinearBoundary: exactly two classes required');
  }
  const sigmaInv = inverse(fit.covariance);
  const w = matvec(sigmaInv, vecSub(mean1, mean2));
  const w0 =
    -0.5 * quadForm(mean1, sigmaInv, mean1) +
    0.5 * quadForm(mean2, sigmaInv, mean2) +
    Math.log(fit.priors[0]! / fit.priors[1]!);
  return { w, w0 };
}
