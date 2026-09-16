import { useMemo } from 'react';
import { evalGrid, linspace, pcg32, smoFitClassifier, sparseLinearKernel, standardNormal, svmDecisionFunction } from '@prml/math';
import { Axes, Heatmap, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const CENTERS: readonly (readonly [number, number])[] = [
  [-2.2, 1.6],
  [2.2, 1.6],
  [0, -2.4],
];
const N_PER_CLASS = 8;
const GRID = linspace(-5, 5, 70);

function cluster(cx: number, cy: number, seed: number) {
  const rng = pcg32(seed);
  return Array.from({ length: N_PER_CLASS }, () => [cx + 0.75 * standardNormal(rng), cy + 0.75 * standardNormal(rng)] as const);
}

export default function MulticlassAmbiguitySvm() {
  const tokens = useResolvedTokens();

  const { clusters, field } = useMemo(() => {
    const clusters = CENTERS.map((c, k) => cluster(c[0], c[1], 20270710 + k));
    const allPoints = clusters.flat();

    const decisions = clusters.map((_, k) => {
      const labels = clusters.flatMap((cl, j) => cl.map(() => (j === k ? (1 as const) : (-1 as const))));
      const fit = smoFitClassifier(allPoints, labels, sparseLinearKernel, { C: 5 });
      return svmDecisionFunction(fit, allPoints, labels, sparseLinearKernel);
    });

    const field = evalGrid(GRID, GRID, (x, y) => {
      const positives = decisions.map((d) => d([x, y]) > 0);
      const count = positives.filter(Boolean).length;
      if (count === 1) return positives.indexOf(true);
      if (count === 0) return -1;
      return -2;
    });
    return { clusters, field };
  }, []);

  const seriesColors = [tokens.series[0]!, tokens.series[1]!, tokens.series[2]!];
  const fill = (v: number) => {
    if (v === -1) return tokens.color.sunken;
    if (v === -2) return tokens.color.danger;
    return seriesColors[v] ?? tokens.color.sunken;
  };

  return (
    <Plot height={280} xDomain={[-5, 5]} yDomain={[-5, 5]} equalAspect label="One-versus-rest regions for three classes, contested and unclaimed areas marked">
      <Heatmap data={field} interpolator={fill} opacity={0.55} />
      <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
      {clusters.map((cl, k) => (
        <ScatterField key={k} points={cl.map((p, i) => ({ x: p[0], y: p[1], id: `${k}-${i}` }))} color={tokens.series[k]} size={3.5} />
      ))}
    </Plot>
  );
}
