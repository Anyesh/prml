import { matmul, pinv } from '../linalg/index.js';
import type { Mat } from '../types.js';

/**
 * PRML 4.16: `Ŵ = X†T`, the multi-output least-squares fit onto 1-of-K target rows.
 * Routed through the SVD-based pseudo-inverse for the same reason as
 * `maximumLikelihoodWeights` in `../regression/basis.ts`: forming the normal equations
 * `XᵀX` squares whatever conditioning the raw design matrix already has, and this chapter's
 * own widget for the least-squares failure mode pushes one cluster of points into a tight,
 * nearly collinear line, which is exactly the shape that makes `XᵀX` far worse conditioned
 * than `X` itself.
 */
export function leastSquaresClassifierWeights(design: Mat, targets: Mat): number[][] {
  return matmul(pinv(design), targets);
}

/** PRML 4.17: `Y = X Ŵ`, one row of scores per input, one column per class. */
export function leastSquaresScores(design: Mat, weights: Mat): number[][] {
  return matmul(design, weights);
}

/** The predicted class: whichever column of a score row is largest. */
export function argmaxIndex(scores: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i]! > scores[best]!) best = i;
  }
  return best;
}
