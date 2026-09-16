import type { Mat, Vec } from '../types.js';
import { logGamma, digamma } from '../special.js';
import { eye, inverse, logDet, matAdd, matScale, matmul, matvec, quadForm, symmetrise, trace, transpose, vecScale } from '../linalg/index.js';

/** `Gam(alpha | a0, b0)` over the weight precision, PRML 10.90. */
export interface VlrPrior {
  readonly a0: number;
  readonly b0: number;
}

/** `q(w) = N(w | mean, cov)` and `q(alpha) = Gam(alpha | a, b)`, PRML 10.94-10.99. */
export interface VlrPosterior {
  readonly mean: Vec;
  readonly cov: Mat;
  readonly a: number;
  readonly b: number;
}

export interface VlrAlphaParams {
  readonly a: number;
  readonly b: number;
}

/**
 * PRML 10.97-10.98: the shape grows by exactly half the weight dimension regardless of the
 * data, because that half-dimension is the same constant every conjugate Gamma-precision
 * update contributes; the rate absorbs `E[w^Tw]`, which mixes how large the fitted weights
 * are with how uncertain `q(w)` still is about them.
 */
export function vlrQAlphaParams(a0: number, b0: number, mean: Vec, cov: Mat): VlrAlphaParams {
  const m = mean.length;
  const eWtW = mean.reduce((s, v) => s + v * v, 0) + trace(cov);
  return { a: a0 + m / 2, b: b0 + 0.5 * eWtW };
}

/**
 * One coordinate-ascent sweep, PRML 10.94-10.99: refits `q(w)` against `current`'s
 * `E[alpha] = a/b`, then refits `q(alpha)` against the new `q(w)`. `vlrFit` below repeats
 * this to convergence; a single call is what a step-through widget advances by.
 */
export function vlrUpdate(design: Mat, targets: Vec, beta: number, prior: VlrPrior, current: VlrAlphaParams): VlrPosterior {
  const eAlpha = current.a / current.b;
  const m = design[0]?.length ?? 0;
  const phiT = transpose(design);
  const precision = symmetrise(matAdd(eye(m, eAlpha), matScale(matmul(phiT, design), beta)));
  const cov = symmetrise(inverse(precision));
  const mean = vecScale(matvec(cov, matvec(phiT, targets)), beta);
  const { a, b } = vlrQAlphaParams(prior.a0, prior.b0, mean, cov);
  return { mean, cov, a, b };
}

export interface VlrFitResult {
  readonly posteriorHistory: readonly VlrPosterior[];
  readonly lowerBoundHistory: readonly number[];
}

export function vlrFit(design: Mat, targets: Vec, beta: number, prior: VlrPrior, iters: number): VlrFitResult {
  const posteriorHistory: VlrPosterior[] = [];
  const lowerBoundHistory: number[] = [];
  let current: VlrAlphaParams = { a: prior.a0, b: prior.b0 };
  for (let i = 0; i < iters; i++) {
    const posterior = vlrUpdate(design, targets, beta, prior, current);
    posteriorHistory.push(posterior);
    lowerBoundHistory.push(vlrLowerBound(design, targets, beta, prior, posterior));
    current = { a: posterior.a, b: posterior.b };
  }
  return { posteriorHistory, lowerBoundHistory };
}

function gaussianEntropy(cov: Mat): number {
  const d = cov.length;
  return 0.5 * logDet(symmetrise(cov)) + 0.5 * d * (1 + Math.log(2 * Math.PI));
}

function gammaEntropy(a: number, b: number): number {
  return a - Math.log(b) + logGamma(a) + (1 - a) * digamma(a);
}

/** `Tr(Phi S Phi^T) = sum_n phi_n^T S phi_n`, the extra data-fit variance from uncertainty in `w`. */
function traceOfPhiSPhiT(design: Mat, cov: Mat): number {
  let sum = 0;
  for (const row of design) sum += quadForm(row, cov, row);
  return sum;
}

/**
 * PRML 10.107, assembled term by term from
 * `E[ln p(t|w)] + E[ln p(w|alpha)] + E[ln p(alpha)] + H[q(w)] + H[q(alpha)]` rather than
 * the book's single combined expression, so each term is checkable against a standard
 * Gaussian or Gamma identity on its own.
 */
export function vlrLowerBound(design: Mat, targets: Vec, beta: number, prior: VlrPrior, posterior: VlrPosterior): number {
  const n = targets.length;
  const d = posterior.mean.length;
  const eAlpha = posterior.a / posterior.b;
  const eLnAlpha = digamma(posterior.a) - Math.log(posterior.b);
  const eWtW = posterior.mean.reduce((s, v) => s + v * v, 0) + trace(posterior.cov);

  const predicted = matvec(design, posterior.mean);
  let residSq = 0;
  for (let i = 0; i < n; i++) {
    const r = targets[i]! - predicted[i]!;
    residSq += r * r;
  }
  const eDataFit = residSq + traceOfPhiSPhiT(design, posterior.cov);
  const termLikelihood = 0.5 * n * Math.log(beta / (2 * Math.PI)) - 0.5 * beta * eDataFit;

  const termPriorW = 0.5 * d * eLnAlpha - 0.5 * d * Math.log(2 * Math.PI) - 0.5 * eAlpha * eWtW;

  const termPriorAlpha = prior.a0 * Math.log(prior.b0) - logGamma(prior.a0) + (prior.a0 - 1) * eLnAlpha - prior.b0 * eAlpha;

  const termEntropyW = gaussianEntropy(posterior.cov);
  const termEntropyAlpha = gammaEntropy(posterior.a, posterior.b);

  return termLikelihood + termPriorW + termPriorAlpha + termEntropyW + termEntropyAlpha;
}
