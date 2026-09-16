import type { Mat, Vec } from '../types.js';

/**
 * The forward-backward (alpha-beta) recursions for a hidden Markov model, PRML 13.2.2 and
 * 13.2.4. This is sum-product specialised to a chain: `hmmForwardUnscaled` is the chain's
 * f-to-z forward message (13.36), `hmmBackwardUnscaled` its root-to-leaf backward message
 * (13.38), and the general factor-graph machinery chapter 8 owns is never invoked, because
 * a chain's messages have this closed two-array form without needing one.
 *
 * Every function here takes `B`, the precomputed emission matrix `p(xn|zn)`, rather than
 * raw observations: the book is explicit (13.2.2) that these recursions do not care what
 * the emission density is, only its value at each of the K states of every zn.
 */

function rowSum(row: Vec): number {
  let total = 0;
  for (const v of row) total += v;
  return total;
}

/** PRML 13.36-13.37: `alpha(zn) = p(x1,...,xn, zn)`. Underflows to zero on a long chain; see `hmmForwardScaled`. */
export function hmmForwardUnscaled(pi: Vec, A: Mat, B: Mat): number[][] {
  const n = B.length;
  const k = pi.length;
  const alpha: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
  for (let j = 0; j < k; j++) alpha[0]![j] = pi[j]! * B[0]![j]!;
  for (let t = 1; t < n; t++) {
    for (let kk = 0; kk < k; kk++) {
      let sum = 0;
      for (let j = 0; j < k; j++) sum += alpha[t - 1]![j]! * A[j]![kk]!;
      alpha[t]![kk] = B[t]![kk]! * sum;
    }
  }
  return alpha;
}

/** PRML 13.38: `beta(zn) = p(xn+1,...,xN | zn)`, with `beta(zN) = 1` (13.39). */
export function hmmBackwardUnscaled(A: Mat, B: Mat): number[][] {
  const n = B.length;
  const k = A.length;
  const beta: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
  beta[n - 1] = new Array(k).fill(1);
  for (let t = n - 2; t >= 0; t--) {
    for (let j = 0; j < k; j++) {
      let sum = 0;
      for (let kk = 0; kk < k; kk++) sum += A[j]![kk]! * B[t + 1]![kk]! * beta[t + 1]![kk]!;
      beta[t]![j] = sum;
    }
  }
  return beta;
}

/** PRML 13.42: `p(X) = sum_zN alpha(zN)`, read off the forward pass's last row. */
export function hmmLikelihoodUnscaled(alpha: Mat): number {
  return rowSum(alpha[alpha.length - 1]!);
}

/** PRML 13.33: the marginal posterior `gamma(zn) = alpha(zn) beta(zn) / p(X)`, normalised row by row. */
export function hmmGammaUnscaled(alpha: Mat, beta: Mat): number[][] {
  return alpha.map((row, t) => {
    const product = row.map((a, k) => a * beta[t]![k]!);
    const total = rowSum(product);
    return product.map((v) => v / total);
  });
}

/**
 * PRML 13.43: the pairwise posterior `xi(zn-1, zn)` for `n = 2, ..., N`, one `K x K`
 * matrix per transition, indexed `0` to `N-2` here (transition into observation `t+1`).
 */
export function hmmXiUnscaled(alpha: Mat, A: Mat, B: Mat, beta: Mat, pX: number): number[][][] {
  const n = alpha.length;
  const k = A.length;
  const out: number[][][] = [];
  for (let t = 1; t < n; t++) {
    const mat: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let j = 0; j < k; j++) {
      for (let kk = 0; kk < k; kk++) {
        mat[j]![kk] = (alpha[t - 1]![j]! * A[j]![kk]! * B[t]![kk]! * beta[t]![kk]!) / pX;
      }
    }
    out.push(mat);
  }
  return out;
}

export interface ScaledForward {
  /** `alpha_hat(zn) = p(zn | x1,...,xn)`, PRML 13.55: a genuine probability distribution over K states at every n. */
  readonly alphaHat: number[][];
  /** `c_n = p(xn | x1,...,xn-1)`, PRML 13.56: the per-step normaliser, and `p(X) = prod c_n` (13.63). */
  readonly c: number[];
}

/**
 * PRML 13.59: the scaled forward recursion. Renormalising `alpha` to a distribution at
 * every step is what keeps it inside machine precision on chains long enough that
 * `hmmForwardUnscaled` would underflow to zero; see the "why scaling exists" test.
 */
export function hmmForwardScaled(pi: Vec, A: Mat, B: Mat): ScaledForward {
  const n = B.length;
  const k = pi.length;
  const alphaHat: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
  const c: number[] = new Array(n).fill(0);

  const raw0 = pi.map((p, j) => p * B[0]![j]!);
  c[0] = rowSum(raw0);
  alphaHat[0] = raw0.map((v) => v / c[0]!);

  for (let t = 1; t < n; t++) {
    const raw: number[] = new Array(k).fill(0);
    for (let kk = 0; kk < k; kk++) {
      let sum = 0;
      for (let j = 0; j < k; j++) sum += alphaHat[t - 1]![j]! * A[j]![kk]!;
      raw[kk] = B[t]![kk]! * sum;
    }
    c[t] = rowSum(raw);
    alphaHat[t] = raw.map((v) => v / c[t]!);
  }

  return { alphaHat, c };
}

/** PRML 13.62: the scaled backward recursion, reusing the `c` computed by `hmmForwardScaled`. */
export function hmmBackwardScaled(A: Mat, B: Mat, c: Vec): number[][] {
  const n = B.length;
  const k = A.length;
  const betaHat: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
  betaHat[n - 1] = new Array(k).fill(1);
  for (let t = n - 2; t >= 0; t--) {
    for (let j = 0; j < k; j++) {
      let sum = 0;
      for (let kk = 0; kk < k; kk++) sum += A[j]![kk]! * B[t + 1]![kk]! * betaHat[t + 1]![kk]!;
      betaHat[t]![j] = sum / c[t + 1]!;
    }
  }
  return betaHat;
}

/** PRML 13.64: `gamma(zn) = alpha_hat(zn) beta_hat(zn)` directly, no further division needed. */
export function hmmGammaScaled(alphaHat: Mat, betaHat: Mat): number[][] {
  return alphaHat.map((row, t) => row.map((a, k) => a * betaHat[t]![k]!));
}

/**
 * PRML 13.65: the scaled pairwise posterior. The book's printed equation reads
 * `xi = c_n alpha_hat(n-1) ... beta_hat(n)`, but substituting the scaling definitions
 * (13.58, 13.60) into the unscaled xi (13.43) gives a division by `c_n`, not a
 * multiplication; the printed "c_n" is a fraction bar lost in typesetting/OCR, and the
 * scaled-vs-unscaled agreement test in forwardBackward.test.ts is what caught it,
 * because a multiplication here still "runs" but silently produces the wrong marginal.
 */
export function hmmXiScaled(alphaHat: Mat, A: Mat, B: Mat, betaHat: Mat, c: Vec): number[][][] {
  const n = alphaHat.length;
  const k = A.length;
  const out: number[][][] = [];
  for (let t = 1; t < n; t++) {
    const mat: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let j = 0; j < k; j++) {
      for (let kk = 0; kk < k; kk++) {
        mat[j]![kk] = (alphaHat[t - 1]![j]! * A[j]![kk]! * B[t]![kk]! * betaHat[t]![kk]!) / c[t]!;
      }
    }
    out.push(mat);
  }
  return out;
}

/** PRML 13.63: `log p(X) = sum_n log c_n`, the numerically safe way to read off the likelihood the scaled recursion never computes directly. */
export function hmmLogLikelihoodScaled(c: Vec): number {
  let total = 0;
  for (const v of c) total += Math.log(v);
  return total;
}
