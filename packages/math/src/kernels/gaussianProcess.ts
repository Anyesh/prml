import {
  cholesky,
  dot,
  eye,
  jitter as addJitter,
  matAdd,
  quadForm,
  solveCholesky,
  symmetrise,
  trace,
  transpose,
  zeros,
} from '../linalg/index.js';
import { mvnSample } from '../distributions/index.js';
import type { Mat, Rng, Vec } from '../types.js';
import { gramMatrix, kernelVector } from './combine.js';
import type { KernelFunction } from './functions.js';

const DEFAULT_JITTER = 1e-8;

export interface GPRegressionModel {
  readonly kernel: KernelFunction;
  readonly trainX: readonly Vec[];
  readonly trainT: Vec;
  readonly noiseVariance: number;
  /** `CN` of PRML 6.62, the Gram matrix plus noise, jittered. */
  readonly covariance: Mat;
  readonly cholesky: Mat;
  /** `CN^-1 t`, computed once and reused by `gpPredict` and both log-likelihood functions. */
  readonly alpha: number[];
}

/**
 * PRML 6.62: `CN = K + beta^-1 I`, factorised once so 6.66, 6.67, 6.69 and 6.70 all reuse the
 * same Cholesky factor rather than re-decomposing an N x N matrix per call.
 *
 * `jitterEps` is added on top of the noise variance regardless of how large that noise
 * already is, because a Gaussian kernel evaluated on closely spaced inputs produces a Gram
 * matrix whose smallest eigenvalues underflow to numerically negative before any noise term
 * is even considered; Cholesky rejects that matrix outright rather than returning a
 * poisoned factor, so the jitter must be added deliberately, not inferred from `noiseVariance`.
 */
export function fitGPRegression(
  kernel: KernelFunction,
  trainX: readonly Vec[],
  trainT: Vec,
  noiseVariance: number,
  jitterEps = DEFAULT_JITTER,
): GPRegressionModel {
  const n = trainX.length;
  const k = gramMatrix(kernel, trainX);
  const covariance = symmetrise(addJitter(matAdd(k, eye(n, noiseVariance)), jitterEps));
  const l = cholesky(covariance);
  const alpha = solveCholesky(l, trainT);
  return { kernel, trainX, trainT, noiseVariance, covariance, cholesky: l, alpha };
}

export interface GPPrediction {
  readonly mean: number;
  readonly variance: number;
}

/** PRML 6.66-6.67, the Gaussian process regression predictive distribution. */
export function gpPredict(model: GPRegressionModel, x: Vec): GPPrediction {
  const k = kernelVector(model.kernel, model.trainX, x);
  const mean = dot(k, model.alpha);
  const cnInvK = solveCholesky(model.cholesky, k);
  const c = model.kernel(x, x) + model.noiseVariance;
  const variance = c - dot(k, cnInvK);
  return { mean, variance };
}

/** PRML 6.69: the log marginal likelihood `ln p(t | theta)`. */
export function gpLogMarginalLikelihood(model: GPRegressionModel): number {
  const n = model.trainT.length;
  let logDet = 0;
  for (let i = 0; i < n; i++) logDet += Math.log(model.cholesky[i]![i]!);
  logDet *= 2;
  const quad = dot(model.trainT, model.alpha);
  return -0.5 * logDet - 0.5 * quad - 0.5 * n * Math.log(2 * Math.PI);
}

/** Solves `CN X = m` column by column, i.e. `CN^-1 m` for a matrix `m`. */
function choSolveMat(l: Mat, m: Mat): number[][] {
  const cols = transpose(m).map((col) => solveCholesky(l, col));
  return transpose(cols);
}

/**
 * PRML 6.70: `d/dtheta_i ln p(t|theta) = -0.5 tr(CN^-1 dCN/dtheta_i) + 0.5 t^T CN^-1
 * dCN/dtheta_i CN^-1 t`. The quadratic term is `alpha^T dCN alpha` rather than the
 * three-matrix product the equation literally shows, because `alpha = CN^-1 t` is already
 * on hand from fitting and reusing it avoids two more N x N solves per hyperparameter.
 *
 * `kernelDerivatives[i]` is `dCN/dtheta_i` as a kernel-shaped function, so `gramMatrix`
 * turns each into the matrix this needs exactly as it turns `kernel` itself into `CN`.
 */
export function gpLogMarginalLikelihoodGradient(
  model: GPRegressionModel,
  kernelDerivatives: readonly KernelFunction[],
): number[] {
  return kernelDerivatives.map((dk) => {
    const dCN = gramMatrix(dk, model.trainX);
    const cnInvDcn = choSolveMat(model.cholesky, dCN);
    const traceTerm = trace(cnInvDcn);
    const quadTerm = quadForm(model.alpha, dCN, model.alpha);
    return -0.5 * traceTerm + 0.5 * quadTerm;
  });
}

/**
 * One draw from the Gaussian process prior `N(0, K)` over `xs`, PRML 6.60, via the existing
 * multivariate-normal sampler. `jitterEps` guards the same near-singular Gram matrix that
 * `fitGPRegression` guards, for callers sampling the prior before any noise term exists.
 */
export function gpPriorSample(rng: Rng, kernel: KernelFunction, xs: readonly Vec[], jitterEps = DEFAULT_JITTER): number[] {
  const cov = addJitter(gramMatrix(kernel, xs), jitterEps);
  return mvnSample(rng, { mean: zeros(xs.length), cov });
}
