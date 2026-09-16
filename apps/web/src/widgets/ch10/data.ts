import { mvnSample, pcg32 } from '@prml/math';

const FAITHFUL_LIKE_SEED = 20261002;

/**
 * A two-cluster, elongated, overlapping dataset standing in for the book's rescaled Old
 * Faithful example (PRML Figures 10.6-10.7), which is copyrighted data this site does not
 * ship. Two real clusters, not the six the widget is asked to fit with, is the point: the
 * gap between six and two is exactly what automatic pruning has to close.
 */
export function faithfulLikeData(): number[][] {
  const rng = pcg32(FAITHFUL_LIKE_SEED);
  const clusters: readonly { mean: readonly [number, number]; cov: number[][] }[] = [
    { mean: [-1.3, -0.8], cov: [[0.55, 0.32], [0.32, 0.42]] },
    { mean: [1.4, 1.0], cov: [[0.42, -0.28], [-0.28, 0.48]] },
  ];
  const perCluster = 45;
  const points: number[][] = [];
  for (const cluster of clusters) {
    for (let i = 0; i < perCluster; i++) {
      points.push(mvnSample(rng, { mean: [...cluster.mean], cov: cluster.cov }));
    }
  }
  return points;
}
