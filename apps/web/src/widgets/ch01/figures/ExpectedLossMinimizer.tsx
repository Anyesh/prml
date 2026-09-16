import { linspace, normalPdf, trapz } from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const T_GRID = linspace(-4, 6, 400);

function conditionalDensity(t: number): number {
  return 0.5 * normalPdf(t, { mu: -1, sigma2: 0.4 }) + 0.5 * normalPdf(t, { mu: 3, sigma2: 0.4 });
}

const DENSITY = T_GRID.map(conditionalDensity);
const CONDITIONAL_MEAN = trapz(
  T_GRID.map((t, i) => t * DENSITY[i]!),
  T_GRID,
);

const Y_GRID = linspace(-4, 6, 200);
const EXPECTED_LOSS = Y_GRID.map((y) =>
  trapz(
    T_GRID.map((t, i) => (y - t) ** 2 * DENSITY[i]!),
    T_GRID,
  ),
);
const MIN_INDEX = EXPECTED_LOSS.reduce((best, v, i) => (v < EXPECTED_LOSS[best]! ? i : best), 0);

export default function ExpectedLossMinimizer() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[-4, 6]} yDomain={[0, 0.5]} label="A two-humped conditional distribution p(t|x)">
        <Axes x={{ label: 't' }} y={{ label: 'p(t|x)' }} grid />
        <Curve points={T_GRID.map((t, i) => [t, DENSITY[i]!] as const)} color={tokens.series[0]!} width={2} />
        <Rule x={CONDITIONAL_MEAN} color={tokens.color.ink} label="E[t|x]" />
      </Plot>
      <Plot height={200} xDomain={[-4, 6]} yDomain={[0, Math.max(...EXPECTED_LOSS)]} label="Expected squared loss against the chosen point prediction y">
        <Axes x={{ label: 'y' }} y={{ label: 'E[L]' }} grid />
        <Curve points={Y_GRID.map((y, i) => [y, EXPECTED_LOSS[i]!] as const)} color={tokens.series[1]!} width={2} />
        <Rule x={Y_GRID[MIN_INDEX]!} color={tokens.color.ink} label="minimiser" />
      </Plot>
      <p className="widget-readout">
        {`The conditional mean is ${CONDITIONAL_MEAN.toFixed(3)}, sitting in the valley between the two humps where the density itself is low; the expected-loss curve is minimised at y=${Y_GRID[MIN_INDEX]!.toFixed(2)}, matching it even though no single value of t near there is likely.`}
      </p>
    </div>
  );
}
