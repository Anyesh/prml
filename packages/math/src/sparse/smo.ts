import type { Mat, Vec } from '../types.js';
import { sparseGramMatrix, type Kernel } from './kernels.js';

export interface BoxQpResult {
  readonly alpha: number[];
  readonly iterations: number;
  readonly converged: boolean;
}

export interface BoxQpProblem {
  /** Signed Gram matrix, `Q[i][j] = y_i y_j K(x_i, x_j)`. Symmetric. */
  readonly q: Mat;
  /** Linear reward coefficients: the solver maximises `sum p_i a_i - (1/2) aᵀQa`. */
  readonly p: Vec;
  /** Equality-constraint coefficients, each `+1` or `-1`: the solver enforces `sum y_i a_i = 0`. */
  readonly y: readonly (1 | -1)[];
  /** Per-index upper bound `C_i`. Must be finite: see the note on hard margins below. */
  readonly upperBound: Vec;
  readonly tol?: number;
  readonly maxIterations?: number;
}

const DEFAULT_TOL = 1e-10;
const DEFAULT_MAX_ITERATIONS = 10_000;
const MIN_CURVATURE = 1e-12;

/**
 * The box-constrained QP every dual in this section reduces to: `max sum p_i a_i -
 * (1/2) aᵀQa` subject to `0 <= a_i <= C_i` and `sum y_i a_i = 0`. PRML 7.10-7.12 is this
 * with `p_i = 1`; PRML 7.32-7.34 is the same problem with a finite `C` in place of the
 * separable case's implicit infinity; and `svr.ts` reduces the epsilon-SVR dual (7.61-7.63)
 * to the same shape by folding each point's `(a_n, a-hat_n)` pair into two signed indices
 * with `y = +1` and `y = -1`. One solver for all three, so a fix to the working-set
 * heuristic below applies everywhere at once instead of drifting across copies.
 *
 * Sequential minimal optimisation (Platt, 1998) solves it by repeatedly choosing the pair
 * of indices with the largest KKT violation and updating only those two analytically: the
 * two-variable restriction of this QP, after eliminating one variable via the equality
 * constraint, is a single-variable quadratic with a closed-form maximiser, so no numerical
 * quadratic programming is ever invoked. Working-set selection is the "maximal violating
 * pair" rule (Keerthi et al., 2001): scan every index once per iteration for the most
 * KKT-violating up/low pair, which is `O(n)` per step and left un-cached because the
 * problems this chapter's widgets pose are small enough that caching would not pay for
 * its own bookkeeping.
 *
 * A hard margin (`C = Infinity`) is not supported: the box update below computes `C_i -
 * C_j`, which is `NaN` when both are infinite. Callers wanting the separable limit pass a
 * large finite `C` (1e6 is enough to make the margin errors invisible at plotting
 * precision) instead of literal `Infinity`.
 */
export function solveBoxConstrainedQp(problem: BoxQpProblem): BoxQpResult {
  const { q, p, y, upperBound } = problem;
  const n = p.length;
  const tol = problem.tol ?? DEFAULT_TOL;
  const maxIterations = problem.maxIterations ?? DEFAULT_MAX_ITERATIONS;

  const alpha = new Array<number>(n).fill(0);
  // Gradient of the minimised form f(a) = (1/2)aᵀQa - pᵀa: grad_i = (Qa)_i - p_i. Starts
  // at -p since alpha is all zero, and is updated incrementally rather than recomputed,
  // which is what keeps each iteration O(n) instead of O(n^2).
  const grad = p.map((pi) => -pi);

  let iterations = 0;
  let converged = false;

  for (; iterations < maxIterations; iterations++) {
    let iUp = -1;
    let iLow = -1;
    let bestUp = -Infinity;
    let bestLow = Infinity;

    for (let t = 0; t < n; t++) {
      const violation = -y[t]! * grad[t]!;
      const canIncrease = y[t]! === 1 ? alpha[t]! < upperBound[t]! : alpha[t]! > 0;
      const canDecrease = y[t]! === 1 ? alpha[t]! > 0 : alpha[t]! < upperBound[t]!;
      if (canIncrease && violation > bestUp) {
        bestUp = violation;
        iUp = t;
      }
      if (canDecrease && violation < bestLow) {
        bestLow = violation;
        iLow = t;
      }
    }

    if (iUp === -1 || iLow === -1 || bestUp - bestLow < tol) {
      converged = true;
      break;
    }

    const i = iUp;
    const j = iLow;
    const yi = y[i]!;
    const yj = y[j]!;
    const qii = q[i]![i]!;
    const qjj = q[j]![j]!;
    const qij = q[i]![j]!;
    const ci = upperBound[i]!;
    const cj = upperBound[j]!;
    const oldAi = alpha[i]!;
    const oldAj = alpha[j]!;
    let newAi: number;
    let newAj: number;

    if (yi !== yj) {
      const curvature = Math.max(qii + qjj + 2 * qij, MIN_CURVATURE);
      const delta = (-grad[i]! - grad[j]!) / curvature;
      const diff = oldAi - oldAj;
      newAi = oldAi + delta;
      newAj = oldAj + delta;
      if (diff > 0) {
        if (newAj < 0) {
          newAj = 0;
          newAi = diff;
        }
      } else if (newAi < 0) {
        newAi = 0;
        newAj = -diff;
      }
      if (diff > ci - cj) {
        if (newAi > ci) {
          newAi = ci;
          newAj = ci - diff;
        }
      } else if (newAj > cj) {
        newAj = cj;
        newAi = cj + diff;
      }
    } else {
      const curvature = Math.max(qii + qjj - 2 * qij, MIN_CURVATURE);
      const delta = (grad[i]! - grad[j]!) / curvature;
      const sum = oldAi + oldAj;
      newAi = oldAi - delta;
      newAj = oldAj + delta;
      if (sum > ci) {
        if (newAi > ci) {
          newAi = ci;
          newAj = sum - ci;
        }
      } else if (newAj < 0) {
        newAj = 0;
        newAi = sum;
      }
      if (sum > cj) {
        if (newAj > cj) {
          newAj = cj;
          newAi = sum - cj;
        }
      } else if (newAi < 0) {
        newAi = 0;
        newAj = sum;
      }
    }

    const dAi = newAi - oldAi;
    const dAj = newAj - oldAj;
    alpha[i] = newAi;
    alpha[j] = newAj;
    for (let t = 0; t < n; t++) {
      grad[t]! += q[t]![i]! * dAi + q[t]![j]! * dAj;
    }
  }

  return { alpha, iterations, converged };
}

export interface SvmFit {
  /** `a_n` of PRML 7.13, one per training point. */
  readonly alpha: number[];
  /** `b` of PRML 7.18/7.37. */
  readonly bias: number;
  /** Indices with `alpha_n` above the support threshold. */
  readonly supportVectors: readonly number[];
  readonly iterations: number;
  readonly converged: boolean;
}

export interface SmoOptions {
  readonly C: number;
  readonly tol?: number;
  readonly maxIterations?: number;
}

/** Below this, an `alpha` is treated as exactly zero: SMO's fixed-point tolerance leaves a residue near machine epsilon, not a genuine support vector. */
const ALPHA_EPS = 1e-8;

/** PRML 7.10-7.18 for the separable case and 7.32-7.37 for the soft margin: the same dual, since a finite `C` just bounds the box. */
export function smoFitClassifier(points: Mat, labels: readonly (1 | -1)[], kernel: Kernel, options: SmoOptions): SvmFit {
  const n = points.length;
  const k = sparseGramMatrix(kernel, points);
  const q = k.map((row, i) => row.map((v, j) => v * labels[i]! * labels[j]!));
  const p = new Array(n).fill(1);
  const upperBound = new Array(n).fill(options.C);

  const { alpha, iterations, converged } = solveBoxConstrainedQp({
    q,
    p,
    y: labels,
    upperBound,
    ...(options.tol !== undefined ? { tol: options.tol } : {}),
    ...(options.maxIterations !== undefined ? { maxIterations: options.maxIterations } : {}),
  });

  // Raw score (no bias yet) at every training point, reused both to pick out the free
  // support vectors and to solve 7.18/7.37 for b at each of them.
  const rawScore = k.map((row) => row.reduce((s, kij, j) => (alpha[j]! > ALPHA_EPS ? s + alpha[j]! * labels[j]! * kij : s), 0));

  const freeEps = ALPHA_EPS * Math.max(1, options.C);
  const free: number[] = [];
  const support: number[] = [];
  for (let i = 0; i < n; i++) {
    if (alpha[i]! > ALPHA_EPS) support.push(i);
    if (alpha[i]! > freeEps && alpha[i]! < options.C - freeEps) free.push(i);
  }

  // 7.18/7.37: average over the numerically stable set of unbounded support vectors. If
  // every support vector saturated at C (a degenerate fit under-regularised for its data),
  // fall back to averaging over all of them rather than leaving b undefined.
  const biasSet = free.length > 0 ? free : support;
  const bias = biasSet.length > 0 ? biasSet.reduce((s, i) => s + (labels[i]! - rawScore[i]!), 0) / biasSet.length : 0;

  return { alpha, bias, supportVectors: support, iterations, converged };
}

/** PRML 7.13: the decision function evaluated at a new point, from a fitted `SvmFit`. */
export function svmDecisionFunction(fit: SvmFit, points: Mat, labels: readonly (1 | -1)[], kernel: Kernel): (x: Vec) => number {
  return (x: Vec) =>
    fit.supportVectors.reduce((s, i) => s + fit.alpha[i]! * labels[i]! * kernel(points[i]!, x), 0) + fit.bias;
}

/** The sign of PRML 7.13. */
export function svmPredict(fit: SvmFit, points: Mat, labels: readonly (1 | -1)[], kernel: Kernel): (x: Vec) => 1 | -1 {
  const decision = svmDecisionFunction(fit, points, labels, kernel);
  return (x: Vec) => (decision(x) >= 0 ? 1 : -1);
}
