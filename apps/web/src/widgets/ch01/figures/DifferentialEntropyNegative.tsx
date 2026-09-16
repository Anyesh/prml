import { differentialEntropyGaussian, linspace } from '@prml/math';
import { Axes, Curve, Rule, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SIGMA2_GRID = linspace(0.001, 3, 300);
const ZERO_CROSSING = 1 / (2 * Math.PI * Math.E);

export default function DifferentialEntropyNegative() {
  const tokens = useResolvedTokens();
  const curve = SIGMA2_GRID.map((s2) => [s2, differentialEntropyGaussian(s2)] as const);

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 3]} yDomain={[-1.5, 2]} label="Differential entropy of a Gaussian against its variance">
        <Axes x={{ label: 'sigma^2' }} y={{ label: 'H[x] (nats)' }} grid zeroLine />
        <Curve points={curve} color={tokens.color.accent} width={2} />
        <Rule x={ZERO_CROSSING} color={tokens.color.danger} label={`1/(2 pi e) = ${ZERO_CROSSING.toFixed(4)}`} />
      </Plot>
      <p className="widget-readout">
        {`H[x] crosses zero at sigma^2 = ${ZERO_CROSSING.toFixed(4)} and goes negative below it: a narrow enough Gaussian has less differential entropy than a single point on a coarse enough grid would, unlike discrete entropy, which never falls below zero.`}
      </p>
    </div>
  );
}
