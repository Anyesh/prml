import type { Rng } from './types.js';
import { NotImplemented } from './types.js';

/**
 * PCG32 (O'Neill 2014), the XSH-RR variant: a 64-bit LCG whose output is permuted down
 * to 32 bits. Chosen over xorshift because distinct `stream` values give provably
 * non-overlapping sequences from one seed, which is what `Rng.fork` needs.
 *
 * Seed and stream accept `number` for call-site convenience; both are widened to 64-bit
 * internally, so values above 2^53 must be passed as `bigint` to survive intact.
 */
export function pcg32(seed: number | bigint, stream: number | bigint = 1): Rng {
  void seed;
  void stream;
  throw new NotImplemented('pcg32');
}

/**
 * Draws `n` values in one pass. Present because sampler inner loops call it per frame
 * and the per-call closure overhead of `next()` shows up at that rate.
 */
export function uniformArray(rng: Rng, n: number): number[] {
  void rng;
  void n;
  throw new NotImplemented('uniformArray');
}

/**
 * Fisher-Yates, returning a new array. The caller's array is not mutated because
 * widgets re-shuffle the same dataset every frame and in-place mutation would make
 * the displayed order depend on frame count.
 */
export function shuffle<T>(rng: Rng, xs: readonly T[]): T[] {
  void rng;
  void xs;
  throw new NotImplemented('shuffle');
}

/**
 * One standard normal deviate. Must use the polar (Marsaglia) form and cache the second
 * deviate against the `Rng` instance, so that consuming exactly one value per call keeps
 * a sampler's output independent of how many samples were drawn before it.
 *
 * Lives beside the RNG rather than in `distributions/normal.ts` because every sampler in
 * the package needs Gaussian noise, and routing them through the normal distribution
 * module would make `linalg` depend on `distributions`.
 */
export function standardNormal(rng: Rng): number {
  void rng;
  throw new NotImplemented('standardNormal');
}

/** Uniform integer in [0, n), free of the modulo bias that `next() * n | 0` introduces. */
export function randInt(rng: Rng, n: number): number {
  void rng;
  void n;
  throw new NotImplemented('randInt');
}

/** Index sampled from `weights`, which need not be normalised but must be non-negative. */
export function categorical(rng: Rng, weights: readonly number[]): number {
  void rng;
  void weights;
  throw new NotImplemented('categorical');
}
