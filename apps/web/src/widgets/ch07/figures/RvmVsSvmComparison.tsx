import { useMemo } from 'react';
import { linspace, pcg32, rvmRegressionFit, rvmRegressionPredictive, smoFitRegression, sparseRbfKernel, standardNormal, svrPredictFunction } from '@prml/math';
import { Axes, Band, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N = 15;
const GAMMA = 2.5;
const GRID = linspace(-1.2, 1.2, 100);

function rbfRow(x: number, centers: readonly number[]): number[] {
  return [1, ...centers.map((c) => Math.exp(-GAMMA * (x - c) ** 2))];
}

export default function RvmVsSvmComparison() {
  const tokens = useResolvedTokens();

  const { xs, ts, rvmCurve, band, rvmRelevant, svmCurve, svmSupport } = useMemo(() => {
    const rng = pcg32(20270740);
    const xs = linspace(-1, 1, N);
    const ts = xs.map((x) => Math.sin(3 * x) + 0.08 * standardNormal(rng));

    const design = xs.map((x) => rbfRow(x, xs));
    const rvm = rvmRegressionFit(design, ts, new Array(design[0]!.length).fill(1), 25);
    const rvmCurve = GRID.map((x) => {
      const phi = rbfRow(x, xs);
      const pred = rvmRegressionPredictive(phi, rvm);
      return { x, mean: pred.mean, sd: Math.sqrt(pred.variance) };
    });
    const rvmRelevant = rvm.relevanceVectors.filter((i) => i > 0).map((i) => i - 1);

    const points = xs.map((x) => [x]);
    const kernel = sparseRbfKernel(GAMMA);
    const svm = smoFitRegression(points, ts, kernel, { C: 8, epsilon: 0.08 });
    const predictSvm = svrPredictFunction(svm, points, kernel);
    const svmCurve = GRID.map((x) => [x, predictSvm([x])] as const);

    return { xs, ts, rvmCurve, band: rvmCurve.map((p) => [p.x, p.mean - p.sd, p.mean + p.sd] as const), rvmRelevant, svmCurve, svmSupport: svm.supportVectors };
  }, []);

  return (
    <Plot height={280} xDomain={[-1.2, 1.2]} yDomain={[-2, 2]} label="RVM (with predictive error bars) against an SVM fitted to the same data">
      <Band points={band} color={tokens.color.accentWash} opacity={0.5} />
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <Curve points={svmCurve} color={tokens.series[1] ?? tokens.color.ink} width={2} dash="dashed" />
      <Curve points={rvmCurve.map((p) => [p.x, p.mean] as const)} color={tokens.color.accent} width={2} />
      <ScatterField points={xs.map((x, i) => ({ x, y: ts[i]!, id: i, color: tokens.color.inkMuted, size: 3 }))} />
      <ScatterField
        points={rvmRelevant.map((i) => ({ x: xs[i]!, y: ts[i]!, id: `rv-${i}`, shape: 'ring' as const, color: tokens.color.accent, size: 7 }))}
      />
      <ScatterField
        points={svmSupport.map((i) => ({ x: xs[i]!, y: ts[i]! - 0.15, id: `sv-${i}`, shape: 'cross' as const, color: tokens.series[1] ?? tokens.color.ink, size: 5 }))}
      />
    </Plot>
  );
}
