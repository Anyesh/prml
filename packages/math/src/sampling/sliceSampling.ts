import type { Rng } from '../types.js';

/** PRML 11.51-11.52: the vertical coordinate of the augmented `(z, u)` pair, in log domain. */
export function sliceVerticalLogLevel(rng: Rng, logTarget: (z: number) => number, z: number): number {
  return logTarget(z) + Math.log(rng.next());
}

export interface SliceInterval {
  readonly lo: number;
  readonly hi: number;
}

/**
 * PRML 11.4's stepping-out procedure: start with a width-`w` window centred on `z0`
 * and grow each side by `w` while it still lies inside the slice. Deterministic given
 * `(z0, logU, w)`, since the book's simplified description (unlike Neal's original,
 * which randomises the initial offset for unbiasedness) places the window
 * symmetrically; the randomness in slice sampling lives entirely in `logU` and in the
 * shrinkage step, not here.
 */
export function sliceSteppingOut(
  z0: number,
  logU: number,
  w: number,
  logTarget: (z: number) => number,
  maxStepsPerSide = 100,
): SliceInterval {
  let lo = z0 - w / 2;
  let hi = z0 + w / 2;
  for (let i = 0; i < maxStepsPerSide && logTarget(lo) > logU; i++) lo -= w;
  for (let i = 0; i < maxStepsPerSide && logTarget(hi) > logU; i++) hi += w;
  return { lo, hi };
}

export interface SliceProposal {
  readonly candidate: number;
  readonly accepted: boolean;
}

export interface SliceShrinkResult {
  readonly z: number;
  readonly proposals: readonly SliceProposal[];
}

/**
 * PRML 11.4's shrinkage procedure: draw uniformly in the current interval, accept if
 * inside the slice, otherwise shrink the interval to that point (keeping `z0` inside)
 * and retry. Guaranteed to terminate because the interval strictly shrinks and `z0`
 * itself is always in the slice by construction.
 */
export function sliceShrink(
  rng: Rng,
  z0: number,
  logU: number,
  interval: SliceInterval,
  logTarget: (z: number) => number,
  maxIterations = 1000,
): SliceShrinkResult {
  let { lo, hi } = interval;
  const proposals: SliceProposal[] = [];
  for (let i = 0; i < maxIterations; i++) {
    const candidate = lo + rng.next() * (hi - lo);
    const accepted = logTarget(candidate) > logU;
    proposals.push({ candidate, accepted });
    if (accepted) return { z: candidate, proposals };
    if (candidate < z0) lo = candidate;
    else hi = candidate;
  }
  throw new Error('sliceShrink: exceeded maxIterations; check logTarget is finite on the interval');
}

export interface SliceStepResult {
  readonly z: number;
  readonly logU: number;
  readonly interval: SliceInterval;
  readonly proposals: readonly SliceProposal[];
}

export function sliceSampleStep(
  rng: Rng,
  z: number,
  logTarget: (z: number) => number,
  w: number,
  maxSteps = 100,
): SliceStepResult {
  const logU = sliceVerticalLogLevel(rng, logTarget, z);
  const interval = sliceSteppingOut(z, logU, w, logTarget, maxSteps);
  const { z: next, proposals } = sliceShrink(rng, z, logU, interval, logTarget, maxSteps);
  return { z: next, logU, interval, proposals };
}

export interface SliceChainResult {
  readonly states: readonly number[];
  readonly steps: readonly SliceStepResult[];
}

export function sliceSampleChain(rng: Rng, initial: number, nSteps: number, logTarget: (z: number) => number, w: number): SliceChainResult {
  const states: number[] = [initial];
  const steps: SliceStepResult[] = [];
  let current = initial;
  for (let i = 0; i < nSteps; i++) {
    const step = sliceSampleStep(rng, current, logTarget, w);
    current = step.z;
    steps.push(step);
    states.push(current);
  }
  return { states, steps };
}
