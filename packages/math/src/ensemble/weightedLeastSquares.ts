import { maximumLikelihoodWeights } from '../regression/basis.js';
import type { Mat, Vec } from '../types.js';

/**
 * Solves `argmin_w sum_n weights[n] (targets[n] - design[n]^T w)^2`, PRML 14.42 written as
 * a weighted normal equation. Rather than forming `Phi^T diag(w) Phi` directly (which
 * squares whatever conditioning `design` already has, same risk as the plain normal
 * equations in `regression/basis.ts`), each row is scaled by `sqrt(weights[n])` first:
 * minimising `||sqrt(W)(t - Phi w)||^2` is exactly the weighted problem, and it lets this
 * reuse the existing SVD-based `maximumLikelihoodWeights` unchanged. This is the one fit
 * routine shared by the mixture of linear regression models (14.5.1, weight = component
 * responsibility) and the regression experts of the mixture of experts (14.5.3).
 */
export function weightedLeastSquaresFit(design: Mat, targets: Vec, weights: Vec): number[] {
  const scaledDesign = design.map((row, n) => {
    const s = Math.sqrt(weights[n]!);
    return row.map((v) => v * s);
  });
  const scaledTargets = targets.map((t, n) => t * Math.sqrt(weights[n]!));
  return maximumLikelihoodWeights(scaledDesign, scaledTargets);
}
