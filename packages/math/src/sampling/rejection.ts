import type { Rng } from '../types.js';
import { gammaPdf } from '../distributions/gamma.js';
import { cauchyPdf, cauchyQuantile, type CauchyParams } from './transform.js';

/** PRML 11.1.2. `targetPdfUnnormalized` need only be proportional to `p(z)` (11.13). */
export interface RejectionSampler {
  readonly sampleProposal: (rng: Rng) => number;
  readonly proposalPdf: (z: number) => number;
  readonly targetPdfUnnormalized: (z: number) => number;
  /** Smallest constant with `k * q(z) >= p(z)` everywhere (11.14 wants it minimal). */
  readonly k: number;
}

export interface RejectionAttempt {
  readonly candidate: number;
  /** Drawn uniformly on `[0, envelope]`, PRML's `u0`. */
  readonly u: number;
  /** `k * q(candidate)`, the comparison function's height at the candidate. */
  readonly envelope: number;
  /** `p(candidate)` (unnormalized). */
  readonly target: number;
  readonly accepted: boolean;
}

/** One draw-and-test cycle of PRML 11.1.2: draw `z0` from `q`, `u0` uniform on `[0, kq(z0)]`, keep if `u0 <= p(z0)`. */
export function rejectionStep(rng: Rng, sampler: RejectionSampler): RejectionAttempt {
  const candidate = sampler.sampleProposal(rng);
  const envelope = sampler.k * sampler.proposalPdf(candidate);
  const u = rng.next() * envelope;
  const target = sampler.targetPdfUnnormalized(candidate);
  return { candidate, u, envelope, target, accepted: u <= target };
}

export interface RejectionResult {
  readonly sample: number;
  readonly attempts: readonly RejectionAttempt[];
}

export function rejectionSample(rng: Rng, sampler: RejectionSampler, maxAttempts = 10000): RejectionResult {
  const attempts: RejectionAttempt[] = [];
  for (let i = 0; i < maxAttempts; i++) {
    const attempt = rejectionStep(rng, sampler);
    attempts.push(attempt);
    if (attempt.accepted) return { sample: attempt.candidate, attempts };
  }
  throw new Error('rejectionSample: exceeded maxAttempts; check that k * q(z) is nowhere less than p(z)');
}

export interface RejectionBatchResult {
  readonly samples: readonly number[];
  readonly trace: readonly RejectionAttempt[];
  /** Empirical fraction accepted, which (11.14) says should tend to `1/k` for a normalized target. */
  readonly acceptanceRate: number;
}

export function rejectionSampleBatch(
  rng: Rng,
  sampler: RejectionSampler,
  n: number,
  maxAttemptsPerSample = 10000,
): RejectionBatchResult {
  const samples: number[] = [];
  const trace: RejectionAttempt[] = [];
  for (let i = 0; i < n; i++) {
    const { sample, attempts } = rejectionSample(rng, sampler, maxAttemptsPerSample);
    samples.push(sample);
    trace.push(...attempts);
  }
  return { samples, trace, acceptanceRate: n / trace.length };
}

export interface GammaCauchyEnvelope extends CauchyParams {
  readonly k: number;
}

/**
 * PRML 11.1.2's worked example: a scaled Cauchy envelope for `Gam(z|a, 1)`, `a > 1`.
 * The book sets the tangency point at the gamma's mode `c = a - 1` and its scale to
 * `b^2 = 2a - 1`, which is the choice that makes that single tangency the global
 * minimum-`k` envelope; `k` itself is then just the height ratio of the two densities
 * at that point.
 */
export function gammaCauchyEnvelope(a: number): GammaCauchyEnvelope {
  if (a <= 1) {
    throw new Error('gammaCauchyEnvelope: requires a > 1, where Gam(z|a, 1) is bell-shaped (PRML 11.1.2)');
  }
  const c = a - 1;
  const b = Math.sqrt(2 * a - 1);
  const target = gammaPdf(c, { shape: a, rate: 1 });
  const proposal = cauchyPdf(c, { b, c });
  return { b, c, k: target / proposal };
}

export function gammaCauchySampler(a: number): RejectionSampler & { envelope: GammaCauchyEnvelope } {
  const envelope = gammaCauchyEnvelope(a);
  return {
    envelope,
    sampleProposal: (rng) => cauchyQuantile(rng.next(), envelope),
    proposalPdf: (z) => cauchyPdf(z, envelope),
    targetPdfUnnormalized: (z) => gammaPdf(z, { shape: a, rate: 1 }),
    k: envelope.k,
  };
}
