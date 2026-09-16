import { kalmanFilter, mvnCovarianceEllipse, mvnMarginal, type LdsParams } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { kalmanTrackingDemo } from '../data.js';

const DEMO = kalmanTrackingDemo();
const SIGMA = 0.6;
export const WORKED_STEP = 3;

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

export function worked() {
  const observations = DEMO.trueTrajectory.map((p, n) => [p[0]! + SIGMA * DEMO.unitNoise[n]![0]!, p[1]! + SIGMA * DEMO.unitNoise[n]![1]!]);
  const steps = kalmanFilter(observations, PARAMS);
  const step = steps[WORKED_STEP]!;
  const predicted = mvnCovarianceEllipse(mvnMarginal({ mean: step.predictedMean, cov: step.predictedCov }, [0, 1]));
  const corrected = mvnCovarianceEllipse(mvnMarginal({ mean: step.mean, cov: step.cov }, [0, 1]));
  return { predicted, corrected };
}

export default function KalmanWorkedStep() {
  const tokens = useResolvedTokens();
  const { predicted, corrected } = worked();

  const bars = [
    { at: 0, value: Math.max(predicted.rx, predicted.ry), color: tokens.series[1]! },
    { at: 1, value: Math.max(corrected.rx, corrected.ry), color: tokens.color.accent },
  ];

  return (
    <Plot width={220} height={180} xDomain={[-0.6, 1.6]} yDomain={[0, Math.max(predicted.rx, predicted.ry) * 1.2]} label="Predicted vs corrected ellipse's longer axis">
      <Axes x={{ label: 'before / after observation' }} y={{ label: 'axis length' }} grid />
      <Bars bars={bars} thickness={0.6} />
    </Plot>
  );
}
