import { mvnSample, pcg32 } from '@prml/math';

const KMEANS_DATA_SEED = 20260910;
const FAITHFUL_DATA_SEED = 20260911;

/**
 * Four blobs arranged in a square, fit with K=3 everywhere this is used. Four groups
 * forced into three clusters means one K-means cluster must always merge two blobs, and
 * which pair merges depends on where the initial means land: exactly the structural
 * ambiguity that makes different seeds land in different local minima. Three
 * well-separated, non-overlapping blobs (one per K) would instead give every seed the
 * same unique optimum, which is not what the seed-sensitivity figures need to show.
 */
export function kmeansDemoData(): number[][] {
  const rng = pcg32(KMEANS_DATA_SEED);
  const centres: readonly [number, number][] = [
    [-2, -2],
    [2, -2],
    [-2, 2],
    [2, 2],
  ];
  const cov = [
    [0.42, 0],
    [0, 0.42],
  ];
  const perCluster = 12;
  const points: number[][] = [];
  for (const centre of centres) {
    for (let i = 0; i < perCluster; i++) {
      points.push(mvnSample(rng, { mean: centre, cov }));
    }
  }
  return points;
}

/**
 * A two-cluster, anisotropic, correlated-within-cluster dataset standing in for the
 * book's Old Faithful example (eruption duration against waiting time), which is
 * copyrighted data this site does not ship. Shape only: elongated, tilted, overlapping
 * at the boundary, which is what makes the hard/soft assignment difference visible.
 */
export function faithfulLikeData(): number[][] {
  const rng = pcg32(FAITHFUL_DATA_SEED);
  const clusters: readonly { mean: readonly [number, number]; cov: number[][] }[] = [
    { mean: [-1.4, -0.9], cov: [[0.7, 0.4], [0.4, 0.55]] },
    { mean: [1.5, 1.1], cov: [[0.55, -0.35], [-0.35, 0.6]] },
  ];
  const perCluster = 26;
  const points: number[][] = [];
  for (const cluster of clusters) {
    for (let i = 0; i < perCluster; i++) {
      points.push(mvnSample(rng, { mean: [...cluster.mean], cov: cluster.cov }));
    }
  }
  return points;
}
