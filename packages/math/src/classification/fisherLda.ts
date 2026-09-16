import { dot, solve } from '../linalg/index.js';
import { vecSub } from '../linalg/index.js';
import { scatterMatrix } from './stats.js';
import type { Mat, Vec } from '../types.js';

/** PRML 4.28: the pooled within-class scatter, summed across the two classes with no normalisation. */
export function withinClassScatter(class1: Mat, class2: Mat, mean1: Vec, mean2: Vec): number[][] {
  const s1 = scatterMatrix(class1, mean1);
  const s2 = scatterMatrix(class2, mean2);
  return s1.map((row, i) => row.map((v, j) => v + s2[i]![j]!));
}

/**
 * PRML 4.30: `w ∝ S_W⁻¹(m2 - m1)`. Solved rather than inverted, because only the direction
 * of `w` is meaningful for the projection and a linear solve is cheaper and better
 * conditioned than forming `S_W⁻¹` explicitly.
 */
export function fisherDirection(mean1: Vec, mean2: Vec, within: Mat): number[] {
  return solve(within, vecSub(mean2, mean1));
}

/** The scalar projections `y = wᵀx` of PRML 4.20, one per row of `points`. */
export function fisherProject(points: Mat, direction: Vec): number[] {
  return points.map((row) => dot(row, direction));
}
