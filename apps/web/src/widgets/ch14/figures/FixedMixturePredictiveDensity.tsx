import { evalGrid, linspace, mixtureLinearRegressionFitEM, normalPdf, type LinearMixtureParams } from '@prml/math';
import { Axes, Heatmap, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import { mixtureRegressionData } from '../data.js';
import '../../widgets.css';

const { design, t } = mixtureRegressionData(50);
const INITIAL: LinearMixtureParams = {
  weights: [
    [0, 1],
    [0, -1],
  ],
  mixing: [0.5, 0.5],
  beta: 10,
};
const FIT = mixtureLinearRegressionFitEM(design, t, INITIAL, 15);
const PARAMS = FIT.paramsHistory[FIT.paramsHistory.length - 1]!;

function predictiveDensity(x: number, tVal: number): number {
  const phi = [1, x];
  let density = 0;
  PARAMS.weights.forEach((w, k) => {
    const mean = w[0]! + w[1]! * phi[1]!;
    density += PARAMS.mixing[k]! * normalPdf(tVal, { mu: mean, sigma2: 1 / PARAMS.beta });
  });
  return density;
}

const X_AXIS = linspace(-1, 1, 50);
const T_AXIS = linspace(-1.5, 1.5, 50);
const GRID = evalGrid(X_AXIS, T_AXIS, predictiveDensity);
const COLOR = sequentialScale([0, Math.max(...GRID.values.flat())]);

export default function FixedMixturePredictiveDensity() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={220} xDomain={[-1, 1]} yDomain={[-1.5, 1.5]} label="Predictive density of a converged mixture of two linear regressions, bimodal at every x">
      <Heatmap data={GRID} interpolator={COLOR} />
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <ScatterField points={design.map((row, i) => ({ x: row[1]!, y: t[i]!, color: tokens.color.plotBg, size: 2 }))} />
    </Plot>
  );
}
