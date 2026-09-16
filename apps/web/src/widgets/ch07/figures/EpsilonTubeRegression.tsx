import { useMemo } from 'react';
import { linspace, pcg32, smoFitRegression, sparseRbfKernel, standardNormal, svrPredictFunction } from '@prml/math';
import { Axes, Band, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N = 16;
const EPSILON = 0.25;
const GAMMA = 0.6;
const GRID = linspace(-3, 3, 120);

function buildData() {
  const rng = pcg32(20270720);
  const xs = linspace(-3, 3, N);
  const ts = xs.map((x) => Math.sin(x) + 0.2 * standardNormal(rng));
  return { xs, ts };
}

export default function EpsilonTubeRegression() {
  const tokens = useResolvedTokens();

  const { xs, ts, curve, band, outside } = useMemo(() => {
    const { xs, ts } = buildData();
    const points = xs.map((x) => [x]);
    const kernel = sparseRbfKernel(GAMMA);
    const fit = smoFitRegression(points, ts, kernel, { C: 4, epsilon: EPSILON });
    const predict = svrPredictFunction(fit, points, kernel);
    const curve = GRID.map((x) => [x, predict([x])] as const);
    const band = GRID.map((x) => {
      const y = predict([x]);
      return [x, y - EPSILON, y + EPSILON] as const;
    });
    const outside = xs.map((x, i) => Math.abs(ts[i]! - predict([x])) > EPSILON + 1e-6);
    return { xs, ts, curve, band, outside };
  }, []);

  return (
    <Plot height={260} xDomain={[-3, 3]} yDomain={[-2, 2]} label="Epsilon-insensitive SVM regression, with points outside the tube marked">
      <Band points={band} color={tokens.color.accentWash} opacity={0.5} />
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <Curve points={curve} color={tokens.color.accent} width={2} />
      <ScatterField
        points={xs.map((x, i) => ({
          x,
          y: ts[i]!,
          id: i,
          color: outside[i] ? tokens.color.danger : tokens.color.ink,
          shape: outside[i] ? 'triangle' : 'circle',
          size: outside[i] ? 4.5 : 3.5,
        }))}
      />
    </Plot>
  );
}
