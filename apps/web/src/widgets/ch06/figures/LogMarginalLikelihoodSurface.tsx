import { compositeKernel, evalGrid, fitGPRegression, gpLogMarginalLikelihood, linspace, pcg32, standardNormal } from '@prml/math';
import { Annotation, Axes, Heatmap, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N = 12;
const rng = pcg32(20260916);
const XS = Array.from({ length: N }, () => rng.next()).sort((a, b) => a - b);
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + 0.15 * standardNormal(rng));
const TRAIN_VECS = XS.map((x) => [x]);
const NOISE_VARIANCE = 1 / 25;

const LOG_THETA0_GRID = linspace(-1.5, 1.5, 40);
const LOG_THETA1_GRID = linspace(-1, 3, 40);

function logMarginalLikelihoodAt(logTheta0: number, logTheta1: number): number {
  const kernel = compositeKernel({ theta0: Math.pow(10, logTheta0), theta1: Math.pow(10, logTheta1), theta2: 0, theta3: 0 });
  const model = fitGPRegression(kernel, TRAIN_VECS, TS, NOISE_VARIANCE);
  return gpLogMarginalLikelihood(model);
}

const FIELD = evalGrid(LOG_THETA0_GRID, LOG_THETA1_GRID, (logTheta0, logTheta1) => logMarginalLikelihoodAt(logTheta0, logTheta1));

let best = { logTheta0: 0, logTheta1: 0, value: -Infinity };
for (let j = 0; j < LOG_THETA1_GRID.length; j++) {
  for (let i = 0; i < LOG_THETA0_GRID.length; i++) {
    const value = FIELD.values[j]![i]!;
    if (value > best.value) best = { logTheta0: LOG_THETA0_GRID[i]!, logTheta1: LOG_THETA1_GRID[j]!, value };
  }
}

export default function LogMarginalLikelihoodSurface() {
  const tokens = useResolvedTokens();
  const minValue = Math.min(...FIELD.values.flat());
  const scale = sequentialScale([minValue, best.value]);

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[-1.5, 1.5]} yDomain={[-1, 3]} label="Log marginal likelihood over log10(theta0) and log10(theta1), grid optimum marked">
        <Heatmap data={FIELD} interpolator={scale} />
        <Axes x={{ label: 'log10(theta0)' }} y={{ label: 'log10(theta1)' }} grid />
        <ScatterField points={[{ x: best.logTheta0, y: best.logTheta1, id: 'optimum' }]} color={tokens.color.danger} size={6} />
        <Annotation x={best.logTheta0} y={best.logTheta1} text="grid optimum" color={tokens.color.danger} dy={-10} plate />
      </Plot>
      <p className="widget-readout">
        {`Twelve points, the same sinusoidal set-up as elsewhere in this chapter. The brightest cell sits at theta0 = ${Math.pow(10, best.logTheta0).toFixed(2)}, theta1 = ${Math.pow(10, best.logTheta1).toFixed(1)}, log marginal likelihood ${best.value.toFixed(2)}: (6.69) scores every setting, and (6.70) is the gradient that would climb this surface directly instead of gridding it.`}
      </p>
    </div>
  );
}
