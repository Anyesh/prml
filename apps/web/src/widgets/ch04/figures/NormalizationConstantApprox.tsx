import { linspace, logGamma } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SHAPES = linspace(1.5, 15, 60);

function laplaceNormalizer(shape: number): number {
  const z0 = shape - 1;
  const precision = 1 / z0;
  const fAtMode = Math.exp((shape - 1) * Math.log(z0) - z0);
  return fAtMode * Math.sqrt((2 * Math.PI) / precision);
}

export default function NormalizationConstantApprox() {
  const tokens = useResolvedTokens();
  const trueZ = SHAPES.map((a) => Math.exp(logGamma(a)));
  const approxZ = SHAPES.map(laplaceNormalizer);
  const ratio = SHAPES.map((_, i) => approxZ[i]! / trueZ[i]!);

  return (
    <Plot height={220} xDomain={[1.5, 15]} yDomain={[0.8, 1.2]} label="Laplace's estimate of the normalising constant, as a fraction of the true value">
      <Axes x={{ label: 'shape a' }} y={{ label: 'Z_Laplace / Z_true' }} grid />
      <Curve points={SHAPES.map((a, i) => [a, ratio[i]!] as const)} color={tokens.color.accent} width={2} />
      <Legend entries={[{ label: 'converges to 1 as a grows', color: tokens.color.accent, mark: 'line' }]} placement="top-right" />
    </Plot>
  );
}
