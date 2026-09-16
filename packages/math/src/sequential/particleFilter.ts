import type { Mat, Rng, Vec } from '../types.js';

/** PRML 13.118: normalises raw per-particle emission likelihoods `p(xn | z_n^(l))` into weights that sum to one. */
export function particleFilterWeights(raw: Vec): number[] {
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((v) => v / total);
}

/**
 * Systematic resampling from one fixed uniform draw `u0` in `[0, 1/L)`: `L` evenly spaced
 * positions `(l + u0) / L` are located in the cumulative weight distribution, which is the
 * standard low-variance alternative to drawing `L` independent categorical samples (each
 * of which could, by chance, miss a low-but-nonzero-weight particle entirely). The last
 * cumulative entry is pinned to exactly 1 to absorb float drift from summing `weights`,
 * so a position of `(L-1+u0)/L` just under 1 is never left unmatched by a cumulative sum
 * that landed at `0.999999999` instead.
 */
export function systematicResampleIndices(u0: number, weights: Vec, L: number): number[] {
  const cumulative: number[] = new Array(weights.length);
  let running = 0;
  for (let i = 0; i < weights.length; i++) {
    running += weights[i]!;
    cumulative[i] = running;
  }
  cumulative[cumulative.length - 1] = 1;

  const indices: number[] = new Array(L);
  let i = 0;
  for (let l = 0; l < L; l++) {
    const position = (l + u0) / L;
    while (cumulative[i]! < position) i++;
    indices[l] = i;
  }
  return indices;
}

/** Draws the single uniform value `systematicResampleIndices` needs from `rng`, so the scheme stays reproducible under a seed. */
export function systematicResample(rng: Rng, weights: Vec, L: number): number[] {
  return systematicResampleIndices(rng.next() / L, weights, L);
}

/**
 * PRML 13.119: propagates a weighted particle set one step, by resampling ancestors in
 * proportion to their current weight and drawing each descendant from the transition
 * kernel `p(z_{n+1} | z_n^(l))`. This is the "bootstrap filter" the book names in 13.3.4;
 * reweighting against the next observation (13.118) is a separate call, since it needs
 * the observation that has not arrived yet at predict time.
 */
export function particleFilterPredict(
  rng: Rng,
  particles: Mat,
  weights: Vec,
  transitionSample: (rng: Rng, z: Vec) => Vec,
): Vec[] {
  const L = particles.length;
  const ancestors = systematicResample(rng, weights, L);
  return ancestors.map((a) => transitionSample(rng, particles[a]!));
}
