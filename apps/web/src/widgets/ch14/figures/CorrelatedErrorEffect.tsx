import { linspace } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const M_VALUES = linspace(1, 30, 60);
const RHOS = [0, 0.3, 0.7];

/** `E_COM / E_AV = rho + (1 - rho) / M`, the variance of an M-average of equal-variance errors at pairwise correlation rho, normalised by a single error's own variance. */
function ratio(m: number, rho: number): number {
  return rho + (1 - rho) / m;
}

export default function CorrelatedErrorEffect() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[1, 30]} yDomain={[0, 1]} label="Predicted committee-to-member error ratio against committee size, at three correlation levels">
      <Axes x={{ label: 'M' }} y={{ label: 'E_COM / E_AV' }} grid />
      {RHOS.map((rho, i) => (
        <Curve key={rho} points={M_VALUES.map((m) => [m, ratio(m, rho)] as const)} color={tokens.series[i]!} width={2} />
      ))}
      <Legend entries={RHOS.map((rho, i) => ({ label: `rho = ${rho}`, color: tokens.series[i]!, mark: 'line' }))} />
    </Plot>
  );
}
