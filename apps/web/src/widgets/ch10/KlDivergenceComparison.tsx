import { useMemo, useState } from 'react';
import {
  bivariateCovariance,
  correlatedPrecision,
  factorizedGaussianForwardKlFit,
  factorizedGaussianReverseKl,
  mvnCovarianceEllipse,
  trueMarginalVariances,
} from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import { ellipseToPolyline } from './lib.js';

import '../widgets.css';

export const title = 'The two failures of a factorised approximation';
export const caption =
  'Drag the correlation. The left panel minimises KL(q||p) and collapses inside the true ellipse; the right minimises KL(p||q) and always reports the true marginal, however elongated it gets.';
export const figure = '10.2';

const MEAN: readonly [number, number] = [0, 0];
const VARIANCES: readonly [number, number] = [1, 1];
const MASS_LEVELS = [0.5, 0.9, 0.99];
const SWEEPS = 40;
const DOMAIN: readonly [number, number] = [-3, 3];

export default function KlDivergenceComparison() {
  const [rho, setRho] = useState(0.85);
  const tokens = useResolvedTokens();

  const p = useMemo(() => correlatedPrecision(MEAN, VARIANCES, rho), [rho]);
  const trueCov = useMemo(() => bivariateCovariance(p), [p]);

  const forward = useMemo(() => {
    const history = factorizedGaussianForwardKlFit(p, { q1: { mu: -1.5, sigma2: 1 }, q2: { mu: 1.5, sigma2: 1 } }, SWEEPS);
    return history[history.length - 1]!;
  }, [p]);

  const reverse = useMemo(() => factorizedGaussianReverseKl(p), [p]);
  const trueVariances = useMemo(() => trueMarginalVariances(p), [p]);

  const truePolylines = MASS_LEVELS.map((mass) => ellipseToPolyline(mvnCovarianceEllipse({ mean: MEAN, cov: trueCov }, mass)));
  const forwardPolylines = MASS_LEVELS.map((mass) =>
    ellipseToPolyline(mvnCovarianceEllipse({ mean: [forward.q1.mu, forward.q2.mu], cov: [[forward.q1.sigma2, 0], [0, forward.q2.sigma2]] }, mass)),
  );
  const reversePolylines = MASS_LEVELS.map((mass) =>
    ellipseToPolyline(mvnCovarianceEllipse({ mean: [reverse.q1.mu, reverse.q2.mu], cov: [[reverse.q1.sigma2, 0], [0, reverse.q2.sigma2]] }, mass)),
  );

  function panel(title: string, qPolylines: (readonly [number, number])[][]) {
    return (
      <Plot width={280} height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label={title}>
        <Axes x={{ label: 'z1' }} y={{ label: 'z2' }} grid />
        {truePolylines.map((pts, i) => (
          <Curve key={`p${i}`} points={pts} color={tokens.color.success} width={1.5} />
        ))}
        {qPolylines.map((pts, i) => (
          <Curve key={`q${i}`} points={pts} color={tokens.color.danger} width={1.5} dash="dashed" />
        ))}
      </Plot>
    );
  }

  return (
    <div>
      <div className="widget-grid">
        {panel('KL(q||p): collapses inside one mode', forwardPolylines)}
        {panel('KL(p||q): matches the true marginal', reversePolylines)}
      </div>
      <Slider
        label="correlation rho"
        value={rho}
        onChange={setRho}
        min={-0.97}
        max={0.97}
        step={0.01}
        format={(v) => v.toFixed(2)}
        hint="Push rho towards 1 and watch the forward-KL ellipse thin out while the reverse-KL one keeps the true shape."
      />
      <p className="widget-readout">
        True marginal variance: {trueVariances[0]!.toFixed(3)}. KL(q||p) reports {forward.q1.sigma2.toFixed(3)}; KL(p||q) reports{' '}
        {reverse.q1.sigma2.toFixed(3)} exactly.
      </p>
    </div>
  );
}
