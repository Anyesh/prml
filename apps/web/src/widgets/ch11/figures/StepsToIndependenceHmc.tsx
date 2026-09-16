import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const MAX_RATIO = 40;

export default function StepsToIndependenceHmc() {
  const tokens = useResolvedTokens();
  const ratios = Array.from({ length: 200 }, (_, i) => 1 + (MAX_RATIO * i) / 199);
  const hmc = ratios.map((r) => [r, r] as const);
  const walk = ratios.map((r) => [r, r * r] as const);

  return (
    <div className="widget-grid">
      <Plot width={420} height={260} xDomain={[1, MAX_RATIO]} yDomain={[1, MAX_RATIO * MAX_RATIO]} yScaleKind="log" label="Evaluations needed for an independent sample, against how elongated the target is">
        <Axes x={{ label: 'sigma_max / sigma_min' }} y={{ label: 'evaluations (log scale)' }} grid />
        <Curve points={hmc} color={tokens.series[0]!} width={2.5} />
        <Curve points={walk} color={tokens.series[1]!} width={2.5} dash="dashed" />
        <Legend
          entries={[
            { label: 'hybrid Monte Carlo, ~ sigma_max/sigma_min', color: tokens.series[0]!, mark: 'line' },
            { label: 'random-walk Metropolis, ~ (sigma_max/sigma_min)^2', color: tokens.series[1]!, mark: 'dashed-line' },
          ]}
        />
      </Plot>
      <p className="widget-readout">
        PRML 11.5.2's closing comparison as a curve: the gap between linear and quadratic scaling widens with every
        unit of extra elongation, and never closes back up.
      </p>
    </div>
  );
}
