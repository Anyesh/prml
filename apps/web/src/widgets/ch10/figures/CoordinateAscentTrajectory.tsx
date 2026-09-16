import { correlatedPrecision, factorizedGaussianForwardKlFit } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const MEAN: readonly [number, number] = [0, 0];
const VARIANCES: readonly [number, number] = [1, 2.2];
const RHO = 0.85;

export default function CoordinateAscentTrajectory() {
  const tokens = useResolvedTokens();
  const p = correlatedPrecision(MEAN, VARIANCES, RHO);
  const trajectory = factorizedGaussianForwardKlFit(p, { q1: { mu: 3, sigma2: 1 }, q2: { mu: -3, sigma2: 1 } }, 12);
  const path = trajectory.map((step) => [step.q1.mu, step.q2.mu] as const);

  return (
    <Plot height={200} xDomain={[-3.5, 3.5]} yDomain={[-3.5, 3.5]} equalAspect label="Coordinate-ascent path of (E[z1], E[z2]) toward the true mean">
      <Axes x={{ label: 'E[z1]' }} y={{ label: 'E[z2]' }} grid zeroLine />
      <Curve points={path} color={tokens.color.accent} width={1.5} />
      <ScatterField
        points={path.map((pt, i) => ({ x: pt[0], y: pt[1], color: i === path.length - 1 ? tokens.color.danger : tokens.color.accent, size: i === 0 || i === path.length - 1 ? 6 : 3 }))}
      />
    </Plot>
  );
}
