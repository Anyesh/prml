import { standardNormal } from '../rng.js';
import type { Family, Mat, Rng, Vec } from '../types.js';
import { eigSym, cholesky, solve, solveCholesky, solveMat } from '../linalg/decompose.js';
import { dot, matmul, matvec, submatrix, subvector, vecAdd, vecSub } from '../linalg/core.js';

export interface MvnParams {
  readonly mean: Vec;
  readonly cov: Mat;
}

export function mvnLogPdf(x: Vec, p: MvnParams): number {
  const d = p.mean.length;
  const diff = vecSub(x, p.mean);
  const l = cholesky(p.cov);
  let logDetCov = 0;
  for (let i = 0; i < d; i++) logDetCov += Math.log(l[i]![i]!);
  logDetCov *= 2;
  const y = solveCholesky(l, diff);
  const quad = dot(diff, y);
  return -0.5 * (d * Math.log(2 * Math.PI) + logDetCov + quad);
}

export function mvnPdf(x: Vec, p: MvnParams): number {
  return Math.exp(mvnLogPdf(x, p));
}

/** Draws via `mean + L z` with `L` the Cholesky factor, so `d` calls to `standardNormal` are consumed per sample. */
export function mvnSample(rng: Rng, p: MvnParams): number[] {
  const l = cholesky(p.cov);
  const z = p.mean.map(() => standardNormal(rng));
  return vecAdd(p.mean, matvec(l, z));
}

/**
 * Marginal over the coordinates in `keep`, which is a plain submatrix selection
 * (PRML 2.98). Order follows `keep`, not ascending index order.
 */
export function mvnMarginal(p: MvnParams, keep: readonly number[]): MvnParams {
  return {
    mean: subvector(p.mean, keep),
    cov: submatrix(p.cov, keep, keep),
  };
}

/**
 * Conditional on the coordinates in `observed` taking the given values (PRML 2.81-2.82).
 * The returned distribution is over the remaining coordinates in ascending index order.
 */
export function mvnConditional(
  p: MvnParams,
  observed: ReadonlyMap<number, number>,
): MvnParams {
  const d = p.mean.length;
  const observedIdx = [...observed.keys()].sort((a, b) => a - b);
  const remainingIdx: number[] = [];
  for (let i = 0; i < d; i++) if (!observed.has(i)) remainingIdx.push(i);

  const xb = observedIdx.map((i) => observed.get(i)!);
  const muA = subvector(p.mean, remainingIdx);
  const muB = subvector(p.mean, observedIdx);
  const saa = submatrix(p.cov, remainingIdx, remainingIdx);
  const sab = submatrix(p.cov, remainingIdx, observedIdx);
  const sba = submatrix(p.cov, observedIdx, remainingIdx);
  const sbb = submatrix(p.cov, observedIdx, observedIdx);

  const diff = vecSub(xb, muB);
  const sbbInvDiff = solve(sbb, diff);
  const mean = vecAdd(muA, matvec(sab, sbbInvDiff));

  const sbbInvSba = solveMat(sbb, sba);
  const correction = matmul(sab, sbbInvSba);
  const cov = saa.map((row, i) => row.map((v, j) => v - correction[i]![j]!));

  return { mean, cov };
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
  const { values, vectors } = eigSym(p.cov);
  const r = Math.sqrt(-2 * Math.log(1 - mass));
  const rx = r * Math.sqrt(values[0]!);
  const ry = r * Math.sqrt(values[1]!);
  const angle = Math.atan2(vectors[0]![1]!, vectors[0]![0]!);
  return { cx: p.mean[0]!, cy: p.mean[1]!, rx, ry, angle };
}

export const Mvn: Family<Vec, MvnParams> = {
  name: 'mvn',
  logPdf: mvnLogPdf,
  pdf: mvnPdf,
  sample: mvnSample,
  mean: (p) => p.mean,
};
