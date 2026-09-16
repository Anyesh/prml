import type { Mat, Vec } from '../types.js';
import { sparseGramMatrix, type Kernel } from './kernels.js';
import { solveBoxConstrainedQp } from './smo.js';

export interface SvrFit {
  /** `a_n - a-hat_n` of PRML 7.57/7.64, one per training point. */
  readonly coefficients: number[];
  /** `b` of PRML 7.69. */
  readonly bias: number;
  /** Indices lying on or outside the epsilon-tube: `|a_n - a-hat_n|` above the support threshold. */
  readonly supportVectors: readonly number[];
  readonly iterations: number;
  readonly converged: boolean;
}

export interface SvrOptions {
  readonly C: number;
  readonly epsilon: number;
  readonly tol?: number;
  readonly maxIterations?: number;
}

const ALPHA_EPS = 1e-8;

/**
 * PRML 7.61-7.69, folded into `solveBoxConstrainedQp`'s shape. Each point contributes two
 * Lagrange multipliers, `a_n` and `a-hat_n` (7.56), which KKT guarantees are never both
 * positive at once (7.65-7.68: their sum being positive would force `epsilon` negative).
 * Laying them out as `2N` signed indices, `a_n` at index `n` with `y = +1` and `a-hat_n`
 * at index `N+n` with `y = -1`, turns 7.61's objective into exactly the classifier's `max
 * sum p_i a_i - (1/2) aᵀQa` with `Q_ij = y_i y_j K(point(i), point(j))` and a per-slot
 * linear term (`t_n - epsilon` for the `a_n` slot, `-t_n - epsilon` for the `a-hat_n`
 * slot) in place of classification's uniform `1`. The equality constraint `sum y_i a_i =
 * 0` is then exactly 7.58. No second SMO implementation is written; this is the same
 * two-variable analytic step doing the same job on a relabelled problem.
 */
export function smoFitRegression(points: Mat, targets: Vec, kernel: Kernel, options: SvrOptions): SvrFit {
  const n = points.length;
  const k = sparseGramMatrix(kernel, points);
  const m = 2 * n;
  const y = new Array<1 | -1>(m);
  const p = new Array<number>(m);
  for (let i = 0; i < n; i++) {
    y[i] = 1;
    p[i] = targets[i]! - options.epsilon;
    y[n + i] = -1;
    p[n + i] = -targets[i]! - options.epsilon;
  }
  const q: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    const ni = i < n ? i : i - n;
    for (let j = 0; j < m; j++) {
      const nj = j < n ? j : j - n;
      q[i]![j] = y[i]! * y[j]! * k[ni]![nj]!;
    }
  }
  const upperBound = new Array(m).fill(options.C);

  const { alpha, iterations, converged } = solveBoxConstrainedQp({
    q,
    p,
    y,
    upperBound,
    ...(options.tol !== undefined ? { tol: options.tol } : {}),
    ...(options.maxIterations !== undefined ? { maxIterations: options.maxIterations } : {}),
  });

  const coefficients = Array.from({ length: n }, (_, i) => alpha[i]! - alpha[n + i]!);
  const rawScore = k.map((row) => row.reduce((s, kij, j) => s + coefficients[j]! * kij, 0));

  // 7.69: b from a point with 0 < a_n < C (upper-boundary point, xi_n = 0), or the
  // symmetric estimate from 0 < a-hat_n < C on the lower boundary. Averaging over both
  // sets is the numerically stable choice the book recommends after 7.69.
  const freeEps = ALPHA_EPS * Math.max(1, options.C);
  const biasEstimates: number[] = [];
  for (let i = 0; i < n; i++) {
    if (alpha[i]! > freeEps && alpha[i]! < options.C - freeEps) {
      biasEstimates.push(targets[i]! - options.epsilon - rawScore[i]!);
    }
    if (alpha[n + i]! > freeEps && alpha[n + i]! < options.C - freeEps) {
      biasEstimates.push(targets[i]! + options.epsilon - rawScore[i]!);
    }
  }
  const bias = biasEstimates.length > 0 ? biasEstimates.reduce((s, v) => s + v, 0) / biasEstimates.length : 0;

  const supportVectors: number[] = [];
  for (let i = 0; i < n; i++) if (Math.abs(coefficients[i]!) > ALPHA_EPS) supportVectors.push(i);

  return { coefficients, bias, supportVectors, iterations, converged };
}

/** PRML 7.64: the regression function evaluated at a new point, from a fitted `SvrFit`. */
export function svrPredictFunction(fit: SvrFit, points: Mat, kernel: Kernel): (x: Vec) => number {
  return (x: Vec) => fit.supportVectors.reduce((s, i) => s + fit.coefficients[i]! * kernel(points[i]!, x), 0) + fit.bias;
}
