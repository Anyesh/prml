import { categorical, mvnSample, pcg32, standardNormal, type Mat } from '@prml/math';

const HMM_DATA_SEED = 20260917;
const KALMAN_NOISE_SEED = 20260919;

export interface HmmDemoComponent {
  readonly mean: readonly [number, number];
  readonly cov: number[][];
}

export interface HmmDemo {
  readonly pi: number[];
  readonly A: number[][];
  readonly components: readonly HmmDemoComponent[];
  readonly data: Mat;
  readonly trueStates: readonly number[];
}

const HMM_PI = [0.5, 0.3, 0.2];
const HMM_A = [
  [0.85, 0.1, 0.05],
  [0.1, 0.8, 0.1],
  [0.1, 0.1, 0.8],
];
const HMM_COMPONENTS: HmmDemoComponent[] = [
  { mean: [0, 0], cov: [[0.5, 0], [0, 0.5]] },
  { mean: [4, 1], cov: [[0.5, 0.15], [0.15, 0.5]] },
  { mean: [1, 4], cov: [[0.5, -0.15], [-0.15, 0.5]] },
];

/**
 * A 3-state Gaussian-emission HMM, sampled generatively as PRML's own figure 13.8 does:
 * an ancestral draw of the state chain, then one Gaussian sample per state. Diagonal
 * transition weight of 0.85 gives visibly persistent runs in one state, exactly what
 * makes a trellis and a Viterbi path worth looking at instead of the data being pure
 * noise switching every step.
 */
export function hmmDemo(length = 9): HmmDemo {
  const rng = pcg32(HMM_DATA_SEED);
  const trueStates: number[] = [];
  const data: number[][] = [];

  let state = categorical(rng, HMM_PI);
  for (let n = 0; n < length; n++) {
    if (n > 0) state = categorical(rng, HMM_A[state]!);
    trueStates.push(state);
    const c = HMM_COMPONENTS[state]!;
    data.push(mvnSample(rng, { mean: [...c.mean], cov: c.cov }));
  }

  return { pi: HMM_PI, A: HMM_A, components: HMM_COMPONENTS, data, trueStates };
}

/**
 * The same 3-state chain as `hmmDemo`, but with wider emission covariances (so each
 * step's density is well under 1) and run long enough that the unscaled forward
 * recursion's row sum genuinely reaches exactly 0 in float64, not merely a very small
 * number. How many steps that takes is read off the actual array at render time, not
 * assumed here; this only has to run long enough for it to have already happened.
 */
export function hmmUnderflowDemo(): HmmDemo {
  const rng = pcg32(HMM_DATA_SEED + 1);
  const length = 260;
  const wideComponents: HmmDemoComponent[] = HMM_COMPONENTS.map((c) => ({
    mean: c.mean,
    cov: c.cov.map((row) => row.map((v) => v * 6)),
  }));
  const trueStates: number[] = [];
  const data: number[][] = [];
  let state = categorical(rng, HMM_PI);
  for (let n = 0; n < length; n++) {
    if (n > 0) state = categorical(rng, HMM_A[state]!);
    trueStates.push(state);
    const c = wideComponents[state]!;
    data.push(mvnSample(rng, { mean: [...c.mean], cov: c.cov }));
  }
  return { pi: HMM_PI, A: HMM_A, components: wideComponents, data, trueStates };
}

export interface KalmanTrackingDemo {
  readonly trueTrajectory: Mat;
  /** Unit Gaussian shocks, one per observation, scaled by the widget's noise slider rather than redrawn on every change. */
  readonly unitNoise: Mat;
}

/**
 * A constant-velocity 2-D trajectory (a gentle curve, not a straight line, so the
 * prediction ellipse's elongation along the heading is visible) plus pre-drawn unit
 * Gaussian shocks. The widget scales `unitNoise` by its slider's sigma to build the
 * actual noisy measurements, so dragging the slider restages the same underlying
 * randomness at a different noise level instead of resampling a new one each time,
 * which would make the filter's response to the slider indistinguishable from ordinary
 * sample-to-sample variation.
 */
export function kalmanTrackingDemo(): KalmanTrackingDemo {
  const N = 12;
  const trueTrajectory: number[][] = [];
  let x = 0;
  let y = 0;
  let angle = 0.15;
  for (let n = 0; n < N; n++) {
    angle += 0.12;
    x += Math.cos(angle) * 1.2;
    y += Math.sin(angle) * 1.2 + 0.3;
    trueTrajectory.push([x, y]);
  }

  const noiseRng = pcg32(KALMAN_NOISE_SEED);
  const unitNoise = trueTrajectory.map(() => [standardNormal(noiseRng), standardNormal(noiseRng)]);

  return { trueTrajectory, unitNoise };
}
