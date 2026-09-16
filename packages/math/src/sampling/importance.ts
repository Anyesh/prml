import { softmax } from '../numeric.js';

/**
 * PRML 11.20's importance ratios `r_l = p(z_l)/q(z_l)`, in log domain so that a target
 * evaluated far in `q`'s tail does not overflow before it is normalized away.
 */
export function importanceLogWeights(
  samples: readonly number[],
  targetLogPdf: (z: number) => number,
  proposalLogPdf: (z: number) => number,
): number[] {
  return samples.map((z) => targetLogPdf(z) - proposalLogPdf(z));
}

/**
 * PRML 11.23: `w_l = r_l / sum_m r_m`. This is exactly a softmax of the log ratios, so
 * it is delegated to the numerically-stable `softmax` already in this package rather
 * than re-deriving the same log-sum-exp shift here.
 */
export function normalizeImportanceWeights(logWeights: readonly number[]): number[] {
  return softmax(logWeights);
}

/** PRML 11.22: the weighted finite-sum estimate of `E[f]`. */
export function importanceEstimate(values: readonly number[], normalizedWeights: readonly number[]): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += normalizedWeights[i]! * values[i]!;
  return sum;
}

/**
 * Not a book equation: the standard diagnostic for the failure PRML 11.1.4 describes in
 * prose, where "the set of importance weights may be dominated by a few weights having
 * large values". `1 / sum(w_l^2)` is `L` when every weight is `1/L` and collapses
 * towards 1 as one weight approaches 1, which is the number the chapter's widget plots
 * to make that collapse visible.
 */
export function effectiveSampleSize(normalizedWeights: readonly number[]): number {
  let sumSq = 0;
  for (const w of normalizedWeights) sumSq += w * w;
  return 1 / sumSq;
}
