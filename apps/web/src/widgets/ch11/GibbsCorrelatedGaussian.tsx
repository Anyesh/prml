import { useMemo, useState } from 'react';
import { gibbsSampleMvn, mvnCovarianceEllipse, pcg32 } from '@prml/math';
import { Axes, Curve, Plot, Trajectory, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

const N_SWEEPS = 25;

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, angle: number, n = 64): (readonly [number, number])[] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle)] as const;
  });
}

export default function GibbsCorrelatedGaussian() {
  const [rho, setRho] = useState(0.9);
  const tokens = useResolvedTokens();

  const mean = [0, 0];
  const cov = [
    [1, rho],
    [rho, 1],
  ];

  const chain = useMemo(() => {
    const rng = pcg32(2026, 80);
    return gibbsSampleMvn(rng, { mean, cov }, [3, -3], N_SWEEPS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rho]);

  const ellipse = mvnCovarianceEllipse({ mean, cov }, 0.9);
  const path: (readonly [number, number])[] = [chain.states[0] as [number, number], ...chain.updates.map((u) => u.z as [number, number])];
  const conditionalVariance = 1 - rho * rho;
  const stepsToIndependence = 1 / conditionalVariance;

  return (
    <div className="widget-grid">
      <Plot width={420} height={340} xDomain={[-4, 4]} yDomain={[-4, 4]} equalAspect label="Gibbs sampling path on a correlated bivariate Gaussian">
        <Axes x={{ label: 'z1' }} y={{ label: 'z2' }} grid zeroLine />
        <Curve points={ellipsePoints(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry, ellipse.angle)} color={tokens.color.inkFaint} width={1.5} />
        <Trajectory path={path} color={tokens.series[0]!} width={1.5} markers fadeOlder />
      </Plot>
      <Slider
        label="Correlation rho"
        value={rho}
        onChange={setRho}
        min={-0.95}
        max={0.95}
        step={0.01}
        hint="Push rho towards 1 and watch the staircase shrink to tiny steps along the ridge."
      />
      <p className="widget-readout">
        Conditional variance 1 - rho^2 = {conditionalVariance.toFixed(3)}: roughly (L/l)^2 ≈{' '}
        {stepsToIndependence.toFixed(1)} sweeps are needed to reach a state independent of the last (PRML 11.3's
        closing estimate).
      </p>
    </div>
  );
}
