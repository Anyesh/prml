import type { Rng, Vec } from '../types.js';
import { mvnConditional, type MvnParams } from '../distributions/mvn.js';
import { normalSample } from '../distributions/normal.js';

/**
 * PRML 11.46-11.48: replace coordinate `index` by a draw from its conditional on every
 * other coordinate. Built on the shared `mvnConditional` (PRML 2.81-2.82) rather than a
 * hand-rolled 2x2 formula, so this generalises past the bivariate case for free and
 * never re-derives the conditional-Gaussian algebra a previous chapter already owns.
 */
export function gibbsMvnCoordinateStep(rng: Rng, z: Vec, params: MvnParams, index: number): number {
  const observed = new Map<number, number>();
  for (let i = 0; i < z.length; i++) if (i !== index) observed.set(i, z[i]!);
  const conditional = mvnConditional(params, observed);
  return normalSample(rng, { mu: conditional.mean[0]!, sigma2: conditional.cov[0]![0]! });
}

export interface GibbsUpdate {
  readonly index: number;
  /** Full state immediately after this one coordinate changed, PRML 11.47's "used straight away". */
  readonly z: readonly number[];
}

/** One full cycle through every coordinate (PRML's Gibbs Sampling box, 11.3), in `order` if given. */
export function gibbsSweep(
  rng: Rng,
  z: Vec,
  params: MvnParams,
  order?: readonly number[],
): { z: number[]; updates: GibbsUpdate[] } {
  const indices = order ?? z.map((_, i) => i);
  let current = [...z];
  const updates: GibbsUpdate[] = [];
  for (const index of indices) {
    const value = gibbsMvnCoordinateStep(rng, current, params, index);
    current = current.slice();
    current[index] = value;
    updates.push({ index, z: current });
  }
  return { z: current, updates };
}

export interface GibbsChainResult {
  /** One entry per sweep, including the initial state at index 0. */
  readonly states: readonly (readonly number[])[];
  /** Every single-coordinate update across every sweep, for the staircase trace (PRML Figure 11.11). */
  readonly updates: readonly GibbsUpdate[];
}

export function gibbsSampleMvn(rng: Rng, params: MvnParams, initial: Vec, nSweeps: number): GibbsChainResult {
  let current: readonly number[] = [...initial];
  const states: (readonly number[])[] = [current];
  const updates: GibbsUpdate[] = [];
  for (let s = 0; s < nSweeps; s++) {
    const sweep = gibbsSweep(rng, current, params);
    current = sweep.z;
    updates.push(...sweep.updates);
    states.push(current);
  }
  return { states, updates };
}

/**
 * PRML 11.50: the over-relaxation step for a Gaussian conditional, replacing the plain
 * Gibbs draw with one biased towards the opposite side of the mean when `alpha < 0`.
 * `alpha = 0` reduces exactly to standard Gibbs.
 */
export function overRelaxationStep(
  zi: number,
  mean: number,
  variance: number,
  alpha: number,
  standardNormalDraw: number,
): number {
  return mean + alpha * (zi - mean) + Math.sqrt(variance) * Math.sqrt(1 - alpha * alpha) * standardNormalDraw;
}
