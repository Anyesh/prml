import type { Mat, Vec } from '../types.js';

export interface ViterbiResult {
  /** `omega(zn) = log p(x1,...,xn, z1,...,zn-1 = argmax, zn)`, PRML 13.68-13.70. */
  readonly omega: number[][];
  /** `psi[n][k]`, the state at `n-1` that maximised the path into state `k` at `n`. Row 0 is unused (filled with -1). */
  readonly psi: number[][];
  /** The single most probable state sequence, PRML 13.71's back-tracked `k*`. */
  readonly path: number[];
  /** `log p(X, Z*)` for the recovered path, the value the final `max` in 13.67 settles on. */
  readonly logProb: number;
}

/**
 * The Viterbi algorithm: max-sum on the HMM chain, in log space (PRML 13.2.5). Log space
 * needs no scaling factors the way `forwardBackward.ts` does, because `max` of logs never
 * accumulates the underflow that a running product of probabilities does.
 *
 * Takes the same `B = p(xn|zn)` emission matrix as `forwardBackward.ts`, so a widget can
 * run both recursions off one precomputed matrix and show where they agree and where the
 * single best path (this) disagrees with the sequence of individually best states
 * (`hmmGammaScaled` maximised pointwise).
 */
export function hmmViterbi(pi: Vec, A: Mat, B: Mat): ViterbiResult {
  const n = B.length;
  const k = pi.length;
  const logA = A.map((row) => row.map((v) => Math.log(v)));
  const logPi = pi.map((v) => Math.log(v));

  const omega: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
  const psi: number[][] = Array.from({ length: n }, () => new Array(k).fill(-1));

  omega[0] = logPi.map((v, j) => v + Math.log(B[0]![j]!));

  for (let t = 1; t < n; t++) {
    for (let kk = 0; kk < k; kk++) {
      let best = -Infinity;
      let bestJ = 0;
      for (let j = 0; j < k; j++) {
        const score = omega[t - 1]![j]! + logA[j]![kk]!;
        if (score > best) {
          best = score;
          bestJ = j;
        }
      }
      psi[t]![kk] = bestJ;
      omega[t]![kk] = Math.log(B[t]![kk]!) + best;
    }
  }

  let finalState = 0;
  let logProb = -Infinity;
  omega[n - 1]!.forEach((v, k2) => {
    if (v > logProb) {
      logProb = v;
      finalState = k2;
    }
  });

  const path: number[] = new Array(n).fill(0);
  path[n - 1] = finalState;
  for (let t = n - 2; t >= 0; t--) {
    path[t] = psi[t + 1]![path[t + 1]!]!;
  }

  return { omega, psi, path, logProb };
}
