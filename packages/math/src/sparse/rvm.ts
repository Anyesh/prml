import {
  diag,
  dot,
  eye,
  inverse,
  matAdd,
  matScale,
  matZeros,
  matmul,
  matvec,
  symmetrise,
  transpose,
  vecSub,
  zeros,
} from '../linalg/index.js';
import { newtonRaphsonLogisticFit } from '../classification/irls.js';
import { predictive as bayesianPredictive, type Predictive, type WeightPosterior } from '../regression/bayesianLinear.js';
import type { Mat, Vec } from '../types.js';

/**
 * An `alpha` at or above this is treated as pruned: PRML says the sparsity mechanism drives
 * a proportion of the hyperparameters to "large (in principle infinite) values" (p.348), and
 * a literal `Infinity` would poison the next iteration's `1/alpha` and `A + beta*PhiᵀPhi`
 * arithmetic with `NaN`. The ceiling is far past where the corresponding weight's posterior
 * variance is indistinguishable from zero at plotting precision.
 */
const ALPHA_MAX = 1e12;

export interface RvmOptions {
  readonly maxIterations?: number;
  readonly tol?: number;
  readonly alphaCeiling?: number;
}

export interface RvmRegressionFit {
  /** `alpha`, PRML 7.87, one per basis function; entries at `alphaCeiling` are pruned. */
  readonly alpha: number[];
  /** `beta`, PRML 7.88. */
  readonly beta: number;
  /** `m`, PRML 7.82. */
  readonly mean: number[];
  /** `Sigma`, PRML 7.83. */
  readonly covariance: number[][];
  /** `A + beta*PhiᵀPhi`, `Sigma`'s inverse; kept so a caller can build a `WeightPosterior` without re-forming it. */
  readonly precision: number[][];
  /** `gamma_i = 1 - alpha_i*Sigma_ii`, PRML 7.89. */
  readonly gamma: number[];
  readonly iterations: number;
  readonly converged: boolean;
  /** Indices whose `alpha` has not reached the ceiling: the surviving basis functions. */
  readonly relevanceVectors: readonly number[];
  /** `alpha` and `mean` after every sweep, so a widget can animate the re-estimation instead of only showing its fixed point. */
  readonly history: readonly { readonly alpha: readonly number[]; readonly mean: readonly number[] }[];
}

/**
 * PRML 7.82-7.89: the direct (non-EM) re-estimation for RVM regression, alternating the
 * weight posterior (7.82-7.83) with the hyperparameter update (7.87-7.89) until both settle.
 * Every `alpha_i` stays finite in this implementation (see `ALPHA_MAX`); the ones the book
 * calls infinite are the ones that reach the ceiling and stay there; their weights are
 * numerically zero rather than symbolically removed from the model, which is the "direct"
 * approach 7.2.1 describes before 7.2.2 replaces it with the faster add/remove algorithm.
 */
export function rvmRegressionFit(design: Mat, targets: Vec, initialAlpha: Vec, initialBeta: number, options?: RvmOptions): RvmRegressionFit {
  const n = design.length;
  const dimension = design[0]?.length ?? 0;
  const maxIterations = options?.maxIterations ?? 1000;
  const tol = options?.tol ?? 1e-12;
  const ceiling = options?.alphaCeiling ?? ALPHA_MAX;

  let alpha = [...initialAlpha];
  let beta = initialBeta;
  let mean = zeros(dimension);
  let covariance = matZeros(dimension, dimension);
  let precision = matZeros(dimension, dimension);
  let gamma = zeros(dimension);
  let iterations = 0;
  let converged = false;
  const history: { alpha: readonly number[]; mean: readonly number[] }[] = [];

  const phiT = transpose(design);
  const phiTt = matvec(phiT, targets);
  const gram = matmul(phiT, design);

  for (; iterations < maxIterations; iterations++) {
    precision = symmetrise(matAdd(diag(alpha), matScale(gram, beta)));
    covariance = symmetrise(inverse(precision));
    mean = matvec(covariance, phiTt).map((v) => v * beta);
    gamma = alpha.map((a, i) => 1 - a * covariance[i]![i]!);
    history.push({ alpha: [...alpha], mean: [...mean] });

    const newAlpha = alpha.map((a, i) => {
      const mSq = mean[i]! * mean[i]!;
      const updated = mSq > 0 ? gamma[i]! / mSq : ceiling;
      return Math.min(updated, ceiling);
    });
    const residual = vecSub(matvec(design, mean), targets);
    const gammaSum = gamma.reduce((s, g) => s + g, 0);
    const newBeta = (n - gammaSum) / dot(residual, residual);

    converged =
      newAlpha.every((a, i) => Math.abs(a - alpha[i]!) <= tol * Math.max(1, Math.min(a, alpha[i]!))) &&
      Math.abs(newBeta - beta) <= tol * Math.max(1, beta);

    alpha = newAlpha;
    beta = newBeta;
    if (converged) break;
  }

  const relevanceVectors = alpha.map((a, i) => i).filter((i) => alpha[i]! < ceiling / 2);
  return { alpha, beta, mean, covariance, precision, gamma, iterations, converged, relevanceVectors, history };
}

/** PRML 7.90-7.91, via the same `predictive` chapter 3 uses: an RVM posterior is a `WeightPosterior` with a per-weight prior instead of a shared one. */
export function rvmRegressionPredictive(phiX: Vec, fit: RvmRegressionFit): Predictive {
  const posterior: WeightPosterior = { mean: fit.mean, cov: fit.covariance, precision: fit.precision };
  return bayesianPredictive(phiX, posterior, fit.beta);
}

export interface SparsityStatistics {
  /** `Q_i`, PRML 7.102. */
  readonly Q: number[];
  /** `S_i`, PRML 7.103. */
  readonly S: number[];
  /** `q_i`, PRML 7.104: the "quality" of basis `i`. */
  readonly q: number[];
  /** `s_i`, PRML 7.105: the "sparsity" of basis `i`. */
  readonly s: number[];
}

/**
 * PRML 7.98-7.107: the quality and sparsity of every basis function at the current
 * hyperparameters, computed from the full `C` of 7.86 via 7.102-7.105 rather than the
 * leave-one-out `C_-i` of 7.98-7.99. Both routes give the same `q_i`, `s_i` (Tipping and
 * Faul, 2003); the full-`C` route needs one `N x N` inverse shared across every basis
 * function instead of `M` leave-one-out ones, which is the version worth teaching even
 * though 7.2.2's sequential algorithm exists precisely to avoid forming `C` at all.
 */
export function rvmSparsityStatistics(design: Mat, targets: Vec, alpha: Vec, beta: number): SparsityStatistics {
  const n = design.length;
  const dimension = design[0]?.length ?? 0;
  const aInv = alpha.map((a) => 1 / a);
  const phiT = transpose(design);
  const scaledPhi = design.map((row) => row.map((v, i) => v * aInv[i]!));
  const c = symmetrise(matAdd(eye(n, 1 / beta), matmul(scaledPhi, phiT)));
  const cInv = inverse(c);
  const cInvT = matvec(cInv, targets);

  const Q = new Array<number>(dimension);
  const S = new Array<number>(dimension);
  const q = new Array<number>(dimension);
  const s = new Array<number>(dimension);
  for (let i = 0; i < dimension; i++) {
    const phiI = phiT[i]!;
    const cInvPhiI = matvec(cInv, phiI);
    Q[i] = dot(phiI, cInvT);
    S[i] = dot(phiI, cInvPhiI);
    const denom = alpha[i]! - S[i]!;
    q[i] = (alpha[i]! * Q[i]!) / denom;
    s[i] = (alpha[i]! * S[i]!) / denom;
  }
  return { Q, S, q, s };
}

/** PRML 7.97: the part of the log marginal likelihood depending on a single `alpha_i`, holding `s_i` and `q_i` fixed. Used to plot Figure 7.11. */
export function rvmSingleAlphaLogEvidenceTerm(alphaCandidate: number, s: number, q: number): number {
  return 0.5 * (Math.log(alphaCandidate) - Math.log(alphaCandidate + s) + (q * q) / (alphaCandidate + s));
}

/** PRML 7.100-7.101: the stationary point of 7.97, or `Infinity` when `q_i^2 <= s_i` and pruning wins. */
export function rvmOptimalSingleAlpha(s: number, q: number): number {
  const qSquared = q * q;
  if (qSquared <= s) return Infinity;
  return (s * s) / (qSquared - s);
}

export interface RvmClassificationFit {
  /** `alpha`, PRML 7.116. */
  readonly alpha: number[];
  /** `w*`, PRML 7.112 (the mode of the Laplace approximation). */
  readonly mean: number[];
  /** `Sigma`, PRML 7.113. */
  readonly covariance: number[][];
  readonly gamma: number[];
  readonly iterations: number;
  readonly converged: boolean;
  readonly relevanceVectors: readonly number[];
}

export interface RvmClassificationOptions extends RvmOptions {
  readonly innerMaxIterations?: number;
  readonly innerTol?: number;
}

/**
 * PRML 7.108-7.119: the RVM's ARD prior applied to logistic regression. `7.109-7.113` is
 * exactly the Bayesian logistic regression MAP problem chapter 4 already solves (compare
 * PRML 4.140-4.144), specialised to a diagonal, per-weight prior precision instead of a
 * shared one; this reuses `newtonRaphsonLogisticFit` from `classification/irls.ts`
 * unchanged rather than re-deriving the same Newton step, exactly as the book's own
 * narrative points back at Section 4.3.3 and 4.5.1 instead of re-deriving IRLS a third
 * time. The outer loop re-estimates `alpha` by 7.116, which is 7.87 with `m_i` replaced
 * by the Laplace mode `w*_i`.
 */
export function rvmClassificationFit(design: Mat, targets: Vec, initialAlpha: Vec, options?: RvmClassificationOptions): RvmClassificationFit {
  const dimension = design[0]?.length ?? 0;
  const maxIterations = options?.maxIterations ?? 200;
  const tol = options?.tol ?? 1e-10;
  const ceiling = options?.alphaCeiling ?? ALPHA_MAX;

  let alpha = [...initialAlpha];
  let mean = zeros(dimension);
  let covariance = matZeros(dimension, dimension);
  let gamma = zeros(dimension);
  let iterations = 0;
  let converged = false;

  for (; iterations < maxIterations; iterations++) {
    const priorPrecision = diag(alpha);
    const fit = newtonRaphsonLogisticFit(design, targets, zeros(dimension), priorPrecision, {
      maxIterations: options?.innerMaxIterations ?? 100,
      tol: options?.innerTol ?? 1e-12,
    });
    mean = fit.weights;
    covariance = symmetrise(inverse(fit.hessian));
    gamma = alpha.map((a, i) => 1 - a * covariance[i]![i]!);

    const newAlpha = alpha.map((a, i) => {
      const wSq = mean[i]! * mean[i]!;
      const updated = wSq > 0 ? gamma[i]! / wSq : ceiling;
      return Math.min(updated, ceiling);
    });

    converged = newAlpha.every((a, i) => Math.abs(a - alpha[i]!) <= tol * Math.max(1, Math.min(a, alpha[i]!)));
    alpha = newAlpha;
    if (converged) break;
  }

  const relevanceVectors = alpha.map((a, i) => i).filter((i) => alpha[i]! < ceiling / 2);
  return { alpha, mean, covariance, gamma, iterations, converged, relevanceVectors };
}
