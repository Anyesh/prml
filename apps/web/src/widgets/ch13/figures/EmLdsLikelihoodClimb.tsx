import { kalmanEmExpectations, kalmanFilter, kalmanLogLikelihood, kalmanMStep, kalmanSmoother, type LdsParams } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { kalmanTrackingDemo } from '../data.js';

const DEMO = kalmanTrackingDemo();
const SIGMA = 0.8;
const OBSERVATIONS = DEMO.trueTrajectory.map((p, n) => [p[0]! + SIGMA * DEMO.unitNoise[n]![0]!, p[1]! + SIGMA * DEMO.unitNoise[n]![1]!]);

const INITIAL: LdsParams = {
  A: [
    [0.9, 0],
    [0, 0.9],
  ],
  Gamma: [
    [1, 0],
    [0, 1],
  ],
  C: [
    [1, 0],
    [0, 1],
  ],
  Sigma: [
    [1, 0],
    [0, 1],
  ],
  mu0: [0, 0],
  V0: [
    [1, 0],
    [0, 1],
  ],
};

const MAX_ITERS = 10;

function runEm(): number[] {
  let params = INITIAL;
  const history: number[] = [];
  for (let i = 0; i < MAX_ITERS; i++) {
    const filtered = kalmanFilter(OBSERVATIONS, params);
    history.push(kalmanLogLikelihood(filtered.map((s) => s.logC)));
    const smoothed = kalmanSmoother(filtered, params);
    const expectations = kalmanEmExpectations(smoothed);
    params = kalmanMStep(OBSERVATIONS, expectations);
  }
  return history;
}

export default function EmLdsLikelihoodClimb() {
  const tokens = useResolvedTokens();
  const history = runEm();
  const points = history.map((ll, i) => [i + 1, ll] as const);

  return (
    <Plot width={340} height={220} xDomain={[1, MAX_ITERS]} yDomain={[Math.min(...history) - 1, Math.max(...history) + 1]} label="LDS log-likelihood across EM iterations">
      <Axes x={{ label: 'EM iteration' }} y={{ label: 'log p(X)' }} grid />
      <Curve points={points} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
