import { useMemo, useState } from 'react';
import { kalmanFilter, mvnCovarianceEllipse, mvnMarginal, type Ellipse, type LdsParams } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider, StepThrough } from '@prml/ui';
import { kalmanTrackingDemo } from './data.js';

import '../widgets.css';

export const title = 'Tracking with a Kalman filter';
export const caption =
  'Drag the sensor noise slider, then step through time. Watch the prediction ellipse (dashed) shrink onto the corrected estimate (solid) when the sensor is trusted, and stay wide, close to the raw measurement, when it is not.';
export const figure = '13.22';

const DEMO = kalmanTrackingDemo();
const N = DEMO.trueTrajectory.length;

function buildParams(sigma: number): LdsParams {
  return {
    A: [
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    Gamma: [
      [0.02, 0, 0, 0],
      [0, 0.02, 0, 0],
      [0, 0, 0.1, 0],
      [0, 0, 0, 0.1],
    ],
    C: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ],
    Sigma: [
      [sigma * sigma, 0],
      [0, sigma * sigma],
    ],
    mu0: [DEMO.trueTrajectory[0]![0]!, DEMO.trueTrajectory[0]![1]!, 1, 0.5],
    V0: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
  };
}

function ellipsePoints(ellipse: Ellipse, segments = 48): (readonly [number, number])[] {
  const points: (readonly [number, number])[] = [];
  const cosA = Math.cos(ellipse.angle);
  const sinA = Math.sin(ellipse.angle);
  for (let i = 0; i <= segments; i++) {
    const t = (2 * Math.PI * i) / segments;
    const ex = ellipse.rx * Math.cos(t);
    const ey = ellipse.ry * Math.sin(t);
    points.push([ellipse.cx + ex * cosA - ey * sinA, ellipse.cy + ex * sinA + ey * cosA]);
  }
  return points;
}

export default function KalmanTracker() {
  const tokens = useResolvedTokens();
  const [sigma, setSigma] = useState(0.6);
  const [step, setStep] = useState(0);

  const observations = useMemo(
    () => DEMO.trueTrajectory.map((p, n) => [p[0]! + sigma * DEMO.unitNoise[n]![0]!, p[1]! + sigma * DEMO.unitNoise[n]![1]!]),
    [sigma],
  );

  const steps = useMemo(() => kalmanFilter(observations, buildParams(sigma)), [observations, sigma]);

  const current = steps[step]!;
  const predictedEllipse = mvnCovarianceEllipse(mvnMarginal({ mean: current.predictedMean, cov: current.predictedCov }, [0, 1]));
  const correctedEllipse = mvnCovarianceEllipse(mvnMarginal({ mean: current.mean, cov: current.cov }, [0, 1]));

  const xs = DEMO.trueTrajectory.map((p) => p[0]!);
  const ys = DEMO.trueTrajectory.map((p) => p[1]!);
  const pad = 3;
  const xDomain: [number, number] = [Math.min(...xs) - pad, Math.max(...xs) + pad];
  const yDomain: [number, number] = [Math.min(...ys) - pad, Math.max(...ys) + pad];

  return (
    <div>
      <Plot width={420} height={340} xDomain={xDomain} yDomain={yDomain} equalAspect label="2-D position tracking">
        <Axes x={{ label: 'x' }} y={{ label: 'y' }} grid />
        <Curve points={DEMO.trueTrajectory.map((p) => [p[0]!, p[1]!] as const)} color={tokens.color.inkFaint} width={1.5} dash="dotted" />
        <Curve points={ellipsePoints(predictedEllipse)} color={tokens.series[1]!} width={1.5} dash="dashed" />
        <Curve points={ellipsePoints(correctedEllipse)} color={tokens.color.accent} width={2} />
        <ScatterField points={DEMO.trueTrajectory.slice(0, step + 1).map((p) => ({ x: p[0]!, y: p[1]!, color: tokens.color.inkFaint, size: 3 }))} />
        <ScatterField
          points={[{ x: observations[step]![0]!, y: observations[step]![1]!, color: tokens.series[1]!, shape: 'cross', size: 6 }]}
          label={() => 'Measurement'}
        />
        <ScatterField
          points={[{ x: current.mean[0]!, y: current.mean[1]!, color: tokens.color.accent, size: 5 }]}
          label={() => 'Corrected estimate'}
        />
      </Plot>
      <Slider label="Measurement noise (sigma)" value={sigma} onChange={setSigma} min={0.05} max={3} step={0.05} />
      <StepThrough step={step} stepCount={N} onStep={setStep} labels={Array.from({ length: N }, (_, t) => `Step ${t + 1}`)} />
      <p className="widget-readout">
        Step {step + 1}: the corrected covariance's largest axis is{' '}
        {Math.max(correctedEllipse.rx, correctedEllipse.ry).toFixed(2)}, against{' '}
        {Math.max(predictedEllipse.rx, predictedEllipse.ry).toFixed(2)} before this observation arrived.
      </p>
    </div>
  );
}
