import type { Family, Mat, Rng, Vec } from '../types.js';
import { NotImplemented } from '../types.js';

export interface MvnParams {
  readonly mean: Vec;
  readonly cov: Mat;
}

export function mvnLogPdf(x: Vec, p: MvnParams): number {
  void x;
  void p;
  throw new NotImplemented('mvnLogPdf');
}

export function mvnPdf(x: Vec, p: MvnParams): number {
  void x;
  void p;
  throw new NotImplemented('mvnPdf');
}

/** Draws via `mean + L z` with `L` the Cholesky factor, so `d` calls to `standardNormal` are consumed per sample. */
export function mvnSample(rng: Rng, p: MvnParams): number[] {
  void rng;
  void p;
  throw new NotImplemented('mvnSample');
}

/**
 * Marginal over the coordinates in `keep`, which is a plain submatrix selection
 * (PRML 2.98). Order follows `keep`, not ascending index order.
 */
export function mvnMarginal(p: MvnParams, keep: readonly number[]): MvnParams {
  void p;
  void keep;
  throw new NotImplemented('mvnMarginal');
}

/**
 * Conditional on the coordinates in `observed` taking the given values (PRML 2.81-2.82).
 * The returned distribution is over the remaining coordinates in ascending index order.
 */
export function mvnConditional(
  p: MvnParams,
  observed: ReadonlyMap<number, number>,
): MvnParams {
  void p;
  void observed;
  throw new NotImplemented('mvnConditional');
}

export interface Ellipse {
  readonly cx: number;
  readonly cy: number;
  readonly rx: number;
  readonly ry: number;
  /** Rotation of the major axis from the x-axis, in radians, counter-clockwise. */
  readonly angle: number;
}

/**
 * The iso-density contour of a 2-D Gaussian enclosing `mass` of the probability, as an
 * ellipse the viz layer can draw directly. `mass` is a probability, not a sigma count,
 * because the chi-squared radius that converts between them differs by dimension and
 * getting it wrong is invisible on screen.
 */
export function mvnCovarianceEllipse(p: MvnParams, mass = 0.95): Ellipse {
  void p;
  void mass;
  throw new NotImplemented('mvnCovarianceEllipse');
}

export const Mvn: Family<Vec, MvnParams> = {
  name: 'mvn',
  logPdf: mvnLogPdf,
  pdf: mvnPdf,
  sample: mvnSample,
  mean: (p) => p.mean,
};
