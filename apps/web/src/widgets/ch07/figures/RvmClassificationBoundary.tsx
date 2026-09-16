import { useMemo } from 'react';
import { evalGrid, laplaceLogisticPredictive, linspace, pcg32, rvmClassificationFit, standardNormal } from '@prml/math';
import { Axes, Heatmap, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N_PER_CLASS = 14;
const GAMMA = 0.8;
const GRID = linspace(-3.5, 3.5, 55);

function rbfRow(x: number, y: number, centers: readonly (readonly [number, number])[]): number[] {
  return [1, ...centers.map((c) => Math.exp(-GAMMA * ((x - c[0]) ** 2 + (y - c[1]) ** 2)))];
}

export default function RvmClassificationBoundary() {
  const tokens = useResolvedTokens();

  const { points, targets, field, relevant } = useMemo(() => {
    const rng = pcg32(20270750);
    const c0 = Array.from({ length: N_PER_CLASS }, () => [-1 + 0.7 * standardNormal(rng), -0.6 + 0.7 * standardNormal(rng)] as const);
    const c1 = Array.from({ length: N_PER_CLASS }, () => [1 + 0.7 * standardNormal(rng), 0.6 + 0.7 * standardNormal(rng)] as const);
    const points = [...c0, ...c1];
    const targets = [...c0.map(() => 0), ...c1.map(() => 1)];
    const design = points.map((p) => rbfRow(p[0], p[1], points));
    const fit = rvmClassificationFit(design, targets, new Array(design[0]!.length).fill(1));

    const posterior = { mean: fit.mean, covariance: fit.covariance };
    const field = evalGrid(GRID, GRID, (x, y) => laplaceLogisticPredictive(rbfRow(x, y, points), posterior));
    const relevant = fit.relevanceVectors.filter((i) => i > 0).map((i) => i - 1);
    return { points, targets, field, relevant };
  }, []);

  const fill = sequentialScale([0, 1]);

  return (
    <Plot height={280} xDomain={[-3.5, 3.5]} yDomain={[-3.5, 3.5]} equalAspect label="RVM classification posterior, with the relevance vectors that survived ringed">
      <Heatmap data={field} interpolator={fill} opacity={0.6} />
      <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
      <ScatterField points={points.map((p, i) => ({ x: p[0], y: p[1], id: i, color: targets[i] === 1 ? tokens.series[1] : tokens.series[0], size: 3.5 }))} />
      <ScatterField
        points={relevant.map((i) => ({ x: points[i]![0], y: points[i]![1], id: `rv-${i}`, shape: 'ring' as const, color: tokens.color.ink, size: 7 }))}
      />
    </Plot>
  );
}
