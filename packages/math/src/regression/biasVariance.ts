import type { Vec } from '../types.js';

export interface BiasVariance {
  /** `(bias)²`, PRML 3.46: how far the average fit sits from the truth. */
  readonly bias2: number;
  /** PRML 3.47: how far individual fits sit from their own average. */
  readonly variance: number;
  /** `bias² + variance`, the part of expected loss a model choice can move. */
  readonly total: number;
}

/**
 * Decomposes expected squared loss over an ensemble of fits, PRML 3.46-3.47.
 *
 * `predictions[l][i]` is dataset `l`'s prediction at test point `i`, and `truth[i]` is the
 * noise-free target there. The intrinsic noise term of 3.41 is deliberately absent: it
 * depends on the data-generating process rather than on the fits, so a function given only
 * fits cannot compute it and must not appear to.
 */
export function biasVarianceDecomposition(
  predictions: readonly (readonly number[])[],
  truth: Vec,
): BiasVariance {
  const avg = ensembleMean(predictions);
  const nPoints = truth.length;
  const nDatasets = predictions.length;

  let bias2 = 0;
  for (let i = 0; i < nPoints; i++) bias2 += (avg[i]! - truth[i]!) ** 2;
  bias2 /= nPoints;

  let variance = 0;
  for (let i = 0; i < nPoints; i++) {
    let spread = 0;
    for (const run of predictions) spread += (run[i]! - avg[i]!) ** 2;
    variance += spread / nDatasets;
  }
  variance /= nPoints;

  return { bias2, variance, total: bias2 + variance };
}

/** The ensemble's average prediction at each test point, `E_D[y(x; D)]` of PRML 3.45. */
export function ensembleMean(predictions: readonly (readonly number[])[]): number[] {
  const nDatasets = predictions.length;
  const nPoints = predictions[0]?.length ?? 0;
  const sum = new Array<number>(nPoints).fill(0);
  for (const run of predictions) {
    for (let i = 0; i < nPoints; i++) sum[i]! += run[i]!;
  }
  return sum.map((s) => s / nDatasets);
}
