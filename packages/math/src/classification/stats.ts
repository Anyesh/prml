import { matAdd, matZeros, outer } from '../linalg/index.js';
import type { Mat, Vec } from '../types.js';

/** Mean of each column across rows of `points`: the class mean μ_k of PRML 4.75-4.76. */
export function columnMeans(points: Mat): number[] {
  const n = points.length;
  const dimension = points[0]?.length ?? 0;
  const sum = new Array<number>(dimension).fill(0);
  for (const row of points) {
    for (let j = 0; j < dimension; j++) sum[j]! += row[j]!;
  }
  return sum.map((s) => s / n);
}

/**
 * Sum of outer products of centred rows, `Σ (xn - mean)(xn - mean)ᵀ`. Left unnormalised
 * because callers disagree on the divisor: Fisher's within-class scatter (PRML 4.28) sums
 * this across classes with no division at all, while the Gaussian generative fit divides by
 * N or Nk to turn it into a covariance (the paragraph following PRML 4.76).
 */
export function scatterMatrix(points: Mat, mean: Vec): number[][] {
  const dimension = mean.length;
  let sum = matZeros(dimension, dimension);
  for (const row of points) {
    const diff = row.map((v, j) => v - mean[j]!);
    sum = matAdd(sum, outer(diff, diff));
  }
  return sum;
}
