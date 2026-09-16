import { categorical } from '../rng.js';
import type { Mat, Rng, Vec } from '../types.js';

/**
 * Log-probability of one path under a homogeneous first-order Markov chain (PRML 13.2):
 * `log pi[s0] + sum_t log A[s(t-1)][s(t)]`. Log space throughout because the chain's
 * probability itself underflows for anything but the shortest paths, exactly the reason
 * the HMM's forward recursion later needs scaling to avoid the same failure.
 */
export function markovChainLogLikelihood(pi: Vec, A: Mat, states: readonly number[]): number {
  if (states.length === 0) return 0;
  let total = Math.log(pi[states[0]!]!);
  for (let t = 1; t < states.length; t++) {
    total += Math.log(A[states[t - 1]!]![states[t]!]!);
  }
  return total;
}

/**
 * Ancestral sampling of a length-`n` path (PRML's generative reading of 13.2): draw the
 * first state from `pi`, then repeatedly draw the next state from the current state's row
 * of `A`. Not golden-fixture-tested because it consumes the RNG stream directly; `rng.ts`'s
 * own fixtures already pin down `categorical`, and what matters here is that the sampled
 * path only ever uses transitions `A` actually allows, checked structurally in the test.
 */
export function markovChainSample(rng: Rng, pi: Vec, A: Mat, n: number): number[] {
  const path: number[] = [];
  let state = categorical(rng, pi);
  path.push(state);
  for (let t = 1; t < n; t++) {
    state = categorical(rng, A[state]!);
    path.push(state);
  }
  return path;
}
