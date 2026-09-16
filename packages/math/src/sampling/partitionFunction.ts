/**
 * PRML 11.72: `Z_E / Z_G ≈ (1/L) sum_l exp(-E(z_l) + G(z_l))`, `{z_l}` drawn from `p_G`.
 * Deterministic given the sample set; the sampler that produced `samplesFromG` (any of
 * this module's others) is where the randomness already happened.
 */
export function partitionFunctionRatioEstimate<Z>(
  samplesFromG: readonly Z[],
  energyE: (z: Z) => number,
  energyG: (z: Z) => number,
): number {
  let sum = 0;
  for (const z of samplesFromG) sum += Math.exp(-energyE(z) + energyG(z));
  return sum / samplesFromG.length;
}

/** PRML 11.74: chaining multiplies a sequence of adjacent ratios into one end-to-end ratio. */
export function chainedPartitionFunctionRatio(ratios: readonly number[]): number {
  return ratios.reduce((product, r) => product * r, 1);
}

/** PRML 11.75: linear interpolation between a tractable energy and the target energy. */
export function interpolatedEnergy<Z>(
  alpha: number,
  energy1: (z: Z) => number,
  energyM: (z: Z) => number,
): (z: Z) => number {
  return (z: Z) => (1 - alpha) * energy1(z) + alpha * energyM(z);
}
