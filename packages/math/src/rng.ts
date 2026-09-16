import type { Rng } from './types.js';

const MASK64 = (1n << 64n) - 1n;
const MULT = 6364136223846793005n;

/**
 * PCG32 (O'Neill 2014), the XSH-RR variant: a 64-bit LCG whose output is permuted down
 * to 32 bits. Chosen over xorshift because distinct `stream` values give provably
 * non-overlapping sequences from one seed, which is what `Rng.fork` needs.
 *
 * Seed and stream accept `number` for call-site convenience; both are widened to 64-bit
 * internally, so values above 2^53 must be passed as `bigint` to survive intact.
 */
export function pcg32(seed: number | bigint, stream: number | bigint = 1): Rng {
  const originalSeed = BigInt(seed);

  function initState(seedB: bigint, streamB: bigint): { state: bigint; inc: bigint } {
    const inc = ((streamB << 1n) | 1n) & MASK64;
    let state = 0n;
    state = (state * MULT + inc) & MASK64;
    state = (state + seedB) & MASK64;
    state = (state * MULT + inc) & MASK64;
    return { state, inc };
  }

  const seeded = initState(originalSeed, BigInt(stream));
  const inc = seeded.inc;
  let state = seeded.state;

  function nextUint32(): number {
    const old = state;
    state = (state * MULT + inc) & MASK64;
    const xorshifted = Number((((old >> 18n) ^ old) >> 27n) & 0xffffffffn);
    const rot = Number(old >> 59n);
    return ((xorshifted >>> rot) | (xorshifted << ((-rot) & 31))) >>> 0;
  }

  function next(): number {
    return nextUint32() / 4294967296;
  }

  function fork(streamId: number): Rng {
    return pcg32(originalSeed, streamId);
  }

  return { nextUint32, next, fork };
}

/**
 * Draws `n` values in one pass. Present because sampler inner loops call it per frame
 * and the per-call closure overhead of `next()` shows up at that rate.
 */
export function uniformArray(rng: Rng, n: number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = rng.next();
  return out;
}

/**
 * Fisher-Yates, returning a new array. The caller's array is not mutated because
 * widgets re-shuffle the same dataset every frame and in-place mutation would make
 * the displayed order depend on frame count.
 */
export function shuffle<T>(rng: Rng, xs: readonly T[]): T[] {
  const out = xs.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = randInt(rng, i + 1);
    const tmp = out[i]!;
    out[i] = out[j]!;
    out[j] = tmp;
  }
  return out;
}

const normalCache = new WeakMap<Rng, number>();

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
  const cached = normalCache.get(rng);
  if (cached !== undefined) {
    normalCache.delete(rng);
    return cached;
  }
  let u = 0;
  let v = 0;
  let s = 0;
  do {
    u = 2 * rng.next() - 1;
    v = 2 * rng.next() - 1;
    s = u * u + v * v;
  } while (s >= 1 || s === 0);
  const factor = Math.sqrt((-2 * Math.log(s)) / s);
  normalCache.set(rng, v * factor);
  return u * factor;
}

/** Uniform integer in [0, n), free of the modulo bias that `next() * n | 0` introduces. */
export function randInt(rng: Rng, n: number): number {
  // Reject draws in the partial final bucket so every outcome in [0, n) stays equally
  // likely; without this, values near 2^32 would be over-represented whenever n does
  // not divide 2^32 evenly.
  const limit = Math.floor(4294967296 / n) * n;
  let x: number;
  do {
    x = rng.nextUint32();
  } while (x >= limit);
  return x % n;
}

/** Index sampled from `weights`, which need not be normalised but must be non-negative. */
export function categorical(rng: Rng, weights: readonly number[]): number {
  let total = 0;
  for (const w of weights) total += w;
  let r = rng.next() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i]!;
    if (r < 0) return i;
  }
  // Floating-point rounding can leave r >= 0 after subtracting every weight;
  // the last index is the only consistent choice left.
  return weights.length - 1;
}
