import type { Vec } from '../types.js';
import { mean as vecMean } from '../numeric.js';

/** Conjugate Gaussian-Gamma prior over `(mu, tau)`, PRML 10.22-10.23. */
export interface NormalGammaPrior {
  readonly mu0: number;
  readonly lambda0: number;
  readonly a0: number;
  readonly b0: number;
}

/** `q(mu) = N(mu | mu, 1/lambda)`, PRML 10.26-10.27. */
export interface QMu {
  readonly mu: number;
  readonly lambda: number;
}

/** `q(tau) = Gam(tau | a, b)`, PRML 10.29-10.30. */
export interface QTau {
  readonly a: number;
  readonly b: number;
}

export interface UnivariateGaussianVbPosterior {
  readonly qMu: QMu;
  readonly qTau: QTau;
}

/**
 * One coordinate-ascent sweep for the conjugate univariate Gaussian (PRML 10.1.3):
 * re-estimates `q(mu)` from the current `E[tau]`, then `q(tau)` from the freshly updated
 * `q(mu)`, in that order, matching the book's own iteration. `qTau.a` is carried through
 * unchanged after the first sweep, since 10.29 fixes it from `N` alone and it never moves
 * again; only `b` and `q(mu)` keep updating.
 */
export function univariateGaussianVbStep(
  data: Vec,
  prior: NormalGammaPrior,
  current: UnivariateGaussianVbPosterior,
): UnivariateGaussianVbPosterior {
  const n = data.length;
  const xbar = vecMean(data);
  const expectedTau = current.qTau.a / current.qTau.b;

  const muN = (prior.lambda0 * prior.mu0 + n * xbar) / (prior.lambda0 + n);
  const lambdaN = (prior.lambda0 + n) * expectedTau;
  const qMu: QMu = { mu: muN, lambda: lambdaN };

  const aN = prior.a0 + n / 2;
  const varMu = 1 / lambdaN;
  let sumSquaredDeviations = 0;
  for (const x of data) sumSquaredDeviations += (x - muN) ** 2 + varMu;
  const priorTerm = prior.lambda0 * ((muN - prior.mu0) ** 2 + varMu);
  const bN = prior.b0 + 0.5 * (sumSquaredDeviations + priorTerm);
  const qTau: QTau = { a: aN, b: bN };

  return { qMu, qTau };
}

/** `univariateGaussianVbStep` iterated `iterations` times, returning `initial` and every sweep after it. */
export function univariateGaussianVbFit(
  data: Vec,
  prior: NormalGammaPrior,
  initial: UnivariateGaussianVbPosterior,
  iterations: number,
): UnivariateGaussianVbPosterior[] {
  const trajectory: UnivariateGaussianVbPosterior[] = [initial];
  let current = initial;
  for (let i = 0; i < iterations; i++) {
    current = univariateGaussianVbStep(data, prior, current);
    trajectory.push(current);
  }
  return trajectory;
}
