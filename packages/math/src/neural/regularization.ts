import type { Vec } from '../types.js';
import { flattenWeights, unflattenWeights } from './network.js';
import type { NetworkSpec, NetworkWeights } from './types.js';

/** PRML 5.112's penalty term, `(λ/2)wᵀw`, over every weight including biases. */
export function weightDecayPenalty(flatWeights: Vec, lambda: number): number {
  let sum = 0;
  for (const w of flatWeights) sum += w * w;
  return 0.5 * lambda * sum;
}

export function weightDecayGradient(flatWeights: Vec, lambda: number): number[] {
  return flatWeights.map((w) => lambda * w);
}

/** PRML 5.112: `Ẽ(w) = E(w) + (λ/2)wᵀw`, given the unregularised error already computed. */
export function regularizedError(rawError: number, weights: NetworkWeights, lambda: number): number {
  return rawError + weightDecayPenalty(flattenWeights(weights), lambda);
}

export function regularizedGradient(spec: NetworkSpec, weights: NetworkWeights, rawGradient: NetworkWeights, lambda: number): NetworkWeights {
  const flatRaw = flattenWeights(rawGradient);
  const flatW = flattenWeights(weights);
  const flatReg = flatRaw.map((g, i) => g + lambda * flatW[i]!);
  return unflattenWeights(spec, flatReg);
}
