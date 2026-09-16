import type { Rng } from '../types.js';
import { categorical } from '../rng.js';
import { importanceLogWeights, normalizeImportanceWeights } from './importance.js';

/** PRML 11.1.5. */
export interface SirOptions {
  readonly sampleProposal: (rng: Rng) => number;
  readonly proposalLogPdf: (z: number) => number;
  readonly targetLogPdf: (z: number) => number;
  /** Size of both the proposal draw and the resampled set, PRML's `L`. */
  readonly l: number;
}

export interface SirResult {
  readonly proposalSamples: readonly number[];
  /** Normalized weights (11.23) the resample step draws from. */
  readonly weights: readonly number[];
  readonly resampled: readonly number[];
}

/**
 * Two stages, PRML 11.1.5: draw `L` samples from `q`, weight them by (11.23), then draw
 * a second set of `L` from the discrete distribution over those samples with
 * probabilities equal to their weights. The result is only approximately `p(z)` for
 * finite `L`; equation 11.26 is the argument that the approximation becomes exact as
 * `L -> infinity`.
 */
export function samplingImportanceResampling(rng: Rng, opts: SirOptions): SirResult {
  const proposalSamples = Array.from({ length: opts.l }, () => opts.sampleProposal(rng));
  const logWeights = importanceLogWeights(proposalSamples, opts.targetLogPdf, opts.proposalLogPdf);
  const weights = normalizeImportanceWeights(logWeights);
  const resampled = Array.from({ length: opts.l }, () => proposalSamples[categorical(rng, weights)]!);
  return { proposalSamples, weights, resampled };
}
