import { bivariateCovariance, correlatedPrecision, factorizedGaussianForwardKlFit, factorizedGaussianReverseKl } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const MEAN: readonly [number, number] = [0, 0];
const VARIANCES: readonly [number, number] = [1, 2.2];
const RHO = 0.85;

export default function VarianceUnderreporting() {
  const tokens = useResolvedTokens();
  const p = correlatedPrecision(MEAN, VARIANCES, RHO);
  const trueCov = bivariateCovariance(p);
  const trajectory = factorizedGaussianForwardKlFit(p, { q1: { mu: 2, sigma2: 1 }, q2: { mu: -2, sigma2: 1 } }, 40);
  const forward = trajectory[trajectory.length - 1]!;
  const reverse = factorizedGaussianReverseKl(p);

  const bars = [
    { at: 0, value: trueCov[0]![0]!, color: tokens.color.inkMuted },
    { at: 1, value: forward.q1.sigma2, color: tokens.color.accent },
    { at: 2, value: reverse.q1.sigma2, color: tokens.color.danger },
  ];

  return (
    <Plot height={180} xDomain={[-0.5, 2.5]} yDomain={[0, Math.max(reverse.q1.sigma2, trueCov[0]![0]!) * 1.15]} label="var(z1) under the true density, KL(q||p), and KL(p||q)">
      <Axes x={{ label: '', ticks: [0, 1, 2], format: (v) => ['true', 'KL(q||p)', 'KL(p||q)'][v] ?? '' }} y={{ label: 'var(z1)' }} grid />
      <Bars bars={bars} thickness={0.6} />
    </Plot>
  );
}
