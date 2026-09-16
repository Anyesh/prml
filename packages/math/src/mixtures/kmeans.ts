import type { Mat, Rng, Vec } from '../types.js';
import { shuffle } from '../rng.js';

export interface KMeansResult {
  readonly means: Mat;
  readonly assignments: readonly number[];
  readonly meansHistory: readonly Mat[];
  readonly assignmentsHistory: readonly (readonly number[])[];
  readonly distortionHistory: readonly number[];
  readonly iterations: number;
  readonly converged: boolean;
}

export function squaredDistance(a: Vec, b: Vec): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    sum += d * d;
  }
  return sum;
}

/**
 * `k` distinct rows of `data`, chosen uniformly without replacement through the seeded
 * PCG32 `Rng` (PRML picks K-means's initial means "at random" without specifying how; a
 * uniform draw from the data itself, rather than from the ambient space, guarantees every
 * centre starts inside the support of the data it will be pulled towards).
 */
export function kmeansInit(rng: Rng, data: Mat, k: number): Mat {
  const indices = shuffle(
    rng,
    data.map((_, i) => i),
  ).slice(0, k);
  return indices.map((i) => [...data[i]!]);
}

/** PRML 9.2: the E step, each point assigned to its nearest mean. Ties keep the lowest index. */
export function kmeansAssign(data: Mat, means: Mat): number[] {
  return data.map((x) => {
    let best = 0;
    let bestDist = squaredDistance(x, means[0]!);
    for (let k = 1; k < means.length; k++) {
      const d = squaredDistance(x, means[k]!);
      if (d < bestDist) {
        bestDist = d;
        best = k;
      }
    }
    return best;
  });
}

/**
 * PRML 9.4: the M step, each mean set to the average of its assigned points. A cluster
 * that is assigned no points keeps its previous mean rather than becoming NaN; the book
 * does not cover this case, and leaving the mean in place is the only choice that does not
 * either crash the next E step or silently move a centre nobody voted for.
 */
export function kmeansUpdateMeans(
  data: Mat,
  assignments: readonly number[],
  k: number,
  previousMeans: Mat,
): Mat {
  const dim = data[0]?.length ?? 0;
  const sums: number[][] = Array.from({ length: k }, () => new Array(dim).fill(0));
  const counts = new Array(k).fill(0);
  data.forEach((x, n) => {
    const c = assignments[n]!;
    counts[c]! += 1;
    for (let d = 0; d < dim; d++) sums[c]![d]! += x[d]!;
  });
  return sums.map((sum, c) => (counts[c]! > 0 ? sum.map((v) => v / counts[c]!) : [...previousMeans[c]!]));
}

/** PRML 9.1: J, the sum of squared distances from each point to its assigned mean. */
export function kmeansDistortion(data: Mat, means: Mat, assignments: readonly number[]): number {
  let total = 0;
  data.forEach((x, n) => {
    total += squaredDistance(x, means[assignments[n]!]!);
  });
  return total;
}

function meansEqual(a: Mat, b: Mat): boolean {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i]!.length; j++) {
      if (a[i]![j]! !== b[i]![j]!) return false;
    }
  }
  return true;
}

/**
 * Alternates the E step and M step from `initialMeans` for exactly `maxIters` rounds,
 * rather than stopping the moment the means settle. A `StepThrough` widget needs a fixed
 * step count known in advance; running on past convergence costs nothing; because the
 * fixed point of assign-then-update is idempotent, every step after `iterations` repeats
 * the same state. Deterministic given its inputs, with the seeded randomness confined to
 * `kmeansInit`, called separately, so this is golden-tested against a fixed starting
 * point with no RNG involved.
 */
export function kmeansFit(data: Mat, initialMeans: Mat, maxIters: number): KMeansResult {
  const k = initialMeans.length;
  let means: Mat = initialMeans.map((row) => [...row]);
  const meansHistory: Mat[] = [means];
  const assignmentsHistory: number[][] = [];
  const distortionHistory: number[] = [];
  let converged = false;
  let iterations = maxIters;

  for (let iter = 0; iter < maxIters; iter++) {
    const assignments = kmeansAssign(data, means);
    assignmentsHistory.push(assignments);
    distortionHistory.push(kmeansDistortion(data, means, assignments));
    const updated = kmeansUpdateMeans(data, assignments, k, means);
    meansHistory.push(updated);
    if (!converged && meansEqual(updated, means)) {
      converged = true;
      iterations = iter + 1;
    }
    means = updated;
  }

  return {
    means,
    assignments: assignmentsHistory[assignmentsHistory.length - 1] ?? [],
    meansHistory,
    assignmentsHistory,
    distortionHistory,
    iterations,
    converged,
  };
}
