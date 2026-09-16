import { kalmanFilter, kalmanSmoother, type LdsParams } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { kalmanTrackingDemo } from '../data.js';

const DEMO = kalmanTrackingDemo();
const SIGMA = 1.1;

const PARAMS: LdsParams = {
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
    [SIGMA * SIGMA, 0],
    [0, SIGMA * SIGMA],
  ],
  mu0: [DEMO.trueTrajectory[0]![0]!, DEMO.trueTrajectory[0]![1]!, 1, 0.5],
  V0: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
};

export default function RtsSmootherComparison() {
  const tokens = useResolvedTokens();
  const observations = DEMO.trueTrajectory.map((p, n) => [p[0]! + SIGMA * DEMO.unitNoise[n]![0]!, p[1]! + SIGMA * DEMO.unitNoise[n]![1]!]);
  const filtered = kalmanFilter(observations, PARAMS);
  const smoothed = kalmanSmoother(filtered, PARAMS);

  const xs = DEMO.trueTrajectory.map((p) => p[0]!);
  const ys = DEMO.trueTrajectory.map((p) => p[1]!);
  const pad = 3;

  return (
    <Plot
      width={360}
      height={300}
      xDomain={[Math.min(...xs) - pad, Math.max(...xs) + pad]}
      yDomain={[Math.min(...ys) - pad, Math.max(...ys) + pad]}
      equalAspect
      label="Filtered against smoothed position estimates"
    >
      <Axes x={{ label: 'x' }} y={{ label: 'y' }} grid />
      <Curve points={DEMO.trueTrajectory.map((p) => [p[0]!, p[1]!] as const)} color={tokens.color.inkFaint} width={1.5} dash="dotted" />
      <Curve points={filtered.map((s) => [s.mean[0]!, s.mean[1]!] as const)} color={tokens.series[1]!} width={2} dash="dashed" />
      <Curve points={smoothed.mean.map((m) => [m[0]!, m[1]!] as const)} color={tokens.color.accent} width={2.5} />
      <ScatterField points={[{ x: DEMO.trueTrajectory[0]![0]!, y: DEMO.trueTrajectory[0]![1]!, color: tokens.color.ink, size: 4 }]} />
    </Plot>
  );
}
