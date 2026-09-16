import { NotImplemented } from './types.js';

/**
 * Lanczos approximation. Accurate to ~1e-13 relative for x > 0, which is the budget the
 * 1e-9 golden tolerance leaves once it propagates through a Dirichlet normaliser.
 */
export function logGamma(x: number): number {
  void x;
  throw new NotImplemented('logGamma');
}

/** Derivative of `logGamma`. Required by variational Bayes, where it appears in every E[ln .] term. */
export function digamma(x: number): number {
  void x;
  throw new NotImplemented('digamma');
}

export function logBeta(a: number, b: number): number {
  void a;
  void b;
  throw new NotImplemented('logBeta');
}

/** `log` of the multivariate gamma function of dimension `d`, the Wishart normaliser. */
export function logMultivariateGamma(x: number, d: number): number {
  void x;
  void d;
  throw new NotImplemented('logMultivariateGamma');
}

export function erf(x: number): number {
  void x;
  throw new NotImplemented('erf');
}

/** Must not be computed as `1 - erf(x)`, which loses all significance for x beyond ~3. */
export function erfc(x: number): number {
  void x;
  throw new NotImplemented('erfc');
}

/** Regularised lower incomplete gamma, `P(a, x)`. Backs the gamma and chi-squared CDFs. */
export function gammaincLower(a: number, x: number): number {
  void a;
  void x;
  throw new NotImplemented('gammaincLower');
}

/** Regularised incomplete beta, `I_x(a, b)`. Backs the beta and Student-t CDFs. */
export function betainc(x: number, a: number, b: number): number {
  void x;
  void a;
  void b;
  throw new NotImplemented('betainc');
}

/** Modified Bessel function of the first kind, order `nu`. The von Mises normaliser. */
export function besselI(nu: number, x: number): number {
  void nu;
  void x;
  throw new NotImplemented('besselI');
}
