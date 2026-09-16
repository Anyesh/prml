import type { Rng } from '../types.js';
import { standardNormal } from '../rng.js';

/** PRML 11.56: `K(r) = ||r||^2 / 2`. */
export function kineticEnergy(r: readonly number[]): number {
  let sum = 0;
  for (const ri of r) sum += ri * ri;
  return 0.5 * sum;
}

/** PRML 11.57: total energy, conserved exactly under the continuous dynamics (11.60) and, for leapfrog, only up to discretisation error. */
export function hamiltonian(z: readonly number[], r: readonly number[], energyFn: (z: readonly number[]) => number): number {
  return energyFn(z) + kineticEnergy(r);
}

export interface PhaseState {
  readonly z: readonly number[];
  readonly r: readonly number[];
}

/**
 * PRML 11.64-11.66: a half-step momentum update, a full-step position update, then a
 * second half-step momentum update. Consecutive calls combine the shared half-steps
 * into what the book calls "leapfrogging"; this function does not fuse them itself, so
 * that `leapfrogTrajectory` can record every intermediate `(z, r)` pair for the
 * chapter's trajectory figure.
 */
export function leapfrogStep(
  state: PhaseState,
  gradEnergyFn: (z: readonly number[]) => readonly number[],
  epsilon: number,
): PhaseState {
  const gradAtZ = gradEnergyFn(state.z);
  const halfR = state.r.map((ri, i) => ri - (epsilon / 2) * gradAtZ[i]!);
  const nextZ = state.z.map((zi, i) => zi + epsilon * halfR[i]!);
  const gradAtNextZ = gradEnergyFn(nextZ);
  const nextR = halfR.map((ri, i) => ri - (epsilon / 2) * gradAtNextZ[i]!);
  return { z: nextZ, r: nextR };
}

/** `L` leapfrog steps from `initial`, returning every intermediate state including the first. */
export function leapfrogTrajectory(
  initial: PhaseState,
  gradEnergyFn: (z: readonly number[]) => readonly number[],
  epsilon: number,
  l: number,
): PhaseState[] {
  const trajectory: PhaseState[] = [initial];
  let state = initial;
  for (let i = 0; i < l; i++) {
    state = leapfrogStep(state, gradEnergyFn, epsilon);
    trajectory.push(state);
  }
  return trajectory;
}

export interface HmcOptions {
  readonly energyFn: (z: readonly number[]) => number;
  readonly gradEnergyFn: (z: readonly number[]) => readonly number[];
  readonly epsilon: number;
  readonly l: number;
}

export interface HmcStepResult {
  readonly zNext: readonly number[];
  readonly accepted: boolean;
  readonly acceptanceProbability: number;
  readonly trajectory: readonly PhaseState[];
}

/**
 * PRML 11.5.2: resample momentum from its Gaussian marginal (a Gibbs step, so it
 * leaves `p(z, r)` invariant on its own), integrate `L` leapfrog steps in a randomly
 * chosen time direction (11.67's proof of detailed balance needs this randomisation so
 * the leapfrog map is its own inverse in expectation), then accept or reject on `H`.
 */
export function hmcStep(rng: Rng, z: readonly number[], opts: HmcOptions): HmcStepResult {
  const r0 = z.map(() => standardNormal(rng));
  const direction = rng.next() < 0.5 ? 1 : -1;
  const trajectory = leapfrogTrajectory({ z, r: r0 }, opts.gradEnergyFn, direction * opts.epsilon, opts.l);
  const final = trajectory[trajectory.length - 1]!;
  const h0 = hamiltonian(z, r0, opts.energyFn);
  const h1 = hamiltonian(final.z, final.r, opts.energyFn);
  const acceptanceProbability = Math.min(1, Math.exp(h0 - h1));
  const accepted = rng.next() < acceptanceProbability;
  return { zNext: accepted ? final.z : z, accepted, acceptanceProbability, trajectory };
}

export interface HmcChainResult {
  readonly states: readonly (readonly number[])[];
  readonly accepted: readonly boolean[];
  readonly acceptanceRate: number;
  readonly trajectories: readonly (readonly PhaseState[])[];
}

export function hmcChain(rng: Rng, initial: readonly number[], nSteps: number, opts: HmcOptions): HmcChainResult {
  const states: (readonly number[])[] = [initial];
  const accepted: boolean[] = [];
  const trajectories: (readonly PhaseState[])[] = [];
  let current = initial;
  for (let i = 0; i < nSteps; i++) {
    const step = hmcStep(rng, current, opts);
    current = step.zNext;
    states.push(current);
    accepted.push(step.accepted);
    trajectories.push(step.trajectory);
  }
  const acceptanceRate = accepted.reduce((s, a) => s + (a ? 1 : 0), 0) / accepted.length;
  return { states, accepted, acceptanceRate, trajectories };
}
