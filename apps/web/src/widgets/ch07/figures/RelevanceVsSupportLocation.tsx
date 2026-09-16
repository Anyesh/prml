import { useMemo } from 'react';
import { evalGrid, linspace, pcg32, rvmClassificationFit, smoFitClassifier, sparseRbfKernel, standardNormal, svmDecisionFunction } from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N_PER_CLASS = 14;
const GAMMA = 0.8;
const GRID = linspace(-3.5, 3.5, 55);

function rbfRow(x: number, y: number, centers: readonly (readonly [number, number])[]): number[] {
  return [1, ...centers.map((c) => Math.exp(-GAMMA * ((x - c[0]) ** 2 + (y - c[1]) ** 2)))];
}

export default function RelevanceVsSupportLocation() {
  const tokens = useResolvedTokens();

  const { points, labels01, field, svmSupport, rvmRelevant } = useMemo(() => {
    const rng = pcg32(20270750);
    const c0 = Array.from({ length: N_PER_CLASS }, () => [-1 + 0.7 * standardNormal(rng), -0.6 + 0.7 * standardNormal(rng)] as const);
    const c1 = Array.from({ length: N_PER_CLASS }, () => [1 + 0.7 * standardNormal(rng), 0.6 + 0.7 * standardNormal(rng)] as const);
    const points = [...c0, ...c1];
    const labels01 = [...c0.map(() => 0), ...c1.map(() => 1)];
    const labelsPm = labels01.map((t) => (t === 1 ? (1 as const) : (-1 as const)));

    const kernel = sparseRbfKernel(GAMMA);
    const svm = smoFitClassifier(points, labelsPm, kernel, { C: 3 });
    const decision = svmDecisionFunction(svm, points, labelsPm, kernel);
    const field = evalGrid(GRID, GRID, (x, y) => decision([x, y]));

    const design = points.map((p) => rbfRow(p[0], p[1], points));
    const rvm = rvmClassificationFit(design, labels01, new Array(design[0]!.length).fill(1));
    const rvmRelevant = rvm.relevanceVectors.filter((i) => i > 0).map((i) => i - 1);

    return { points, labels01, field, svmSupport: svm.supportVectors, rvmRelevant };
  }, []);

  return (
    <Plot height={280} xDomain={[-3.5, 3.5]} yDomain={[-3.5, 3.5]} equalAspect label="Support vectors against relevance vectors on the same boundary">
      <ContourField data={field} levels={[0]} color={tokens.color.ink} lineWidth={2} />
      <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
      <ScatterField points={points.map((p, i) => ({ x: p[0], y: p[1], id: i, color: labels01[i] === 1 ? tokens.series[1] : tokens.series[0], size: 3 }))} />
      <ScatterField
        points={svmSupport.map((i) => ({ x: points[i]![0], y: points[i]![1], id: `sv-${i}`, shape: 'cross' as const, color: tokens.color.danger, size: 6 }))}
      />
      <ScatterField
        points={rvmRelevant.map((i) => ({ x: points[i]![0], y: points[i]![1], id: `rv-${i}`, shape: 'ring' as const, color: tokens.color.accent, size: 8 }))}
      />
    </Plot>
  );
}
