import type { Rng } from '../types.js';
import { standardNormal } from '../rng.js';

/**
 * PRML 11.44 in log domain (numerically safer than exponentiating the ratio first).
 * With a symmetric proposal the two `logQ` terms are equal for any `q`, so passing `0`
 * for both collapses this to the basic Metropolis criterion (11.33), exactly as the
 * book notes just below (11.44).
 */
export function mhAcceptanceLogRatio(
  candidateLogTarget: number,
  currentLogTarget: number,
  logQCurrentGivenCandidate: number,
  logQCandidateGivenCurrent: number,
): number {
  return candidateLogTarget - currentLogTarget + logQCurrentGivenCandidate - logQCandidateGivenCurrent;
}

export function mhAcceptanceProbability(logRatio: number): number {
  return Math.min(1, Math.exp(logRatio));
}

export interface MhSampler<S> {
  readonly proposalSample: (rng: Rng, current: S) => S;
  /** `log q(to | from)`. Pass a function returning `0` for a symmetric proposal. */
  readonly proposalLogPdf: (to: S, from: S) => number;
  readonly targetLogPdf: (z: S) => number;
}

export interface MhStepResult<S> {
  readonly next: S;
  readonly candidate: S;
  readonly accepted: boolean;
  readonly acceptanceProbability: number;
}

/** One cycle of PRML 11.2.2: propose, accept with probability (11.44), else stay put. */
export function metropolisHastingsStep<S>(rng: Rng, current: S, sampler: MhSampler<S>): MhStepResult<S> {
  const candidate = sampler.proposalSample(rng, current);
  const logRatio = mhAcceptanceLogRatio(
    sampler.targetLogPdf(candidate),
    sampler.targetLogPdf(current),
    sampler.proposalLogPdf(current, candidate),
    sampler.proposalLogPdf(candidate, current),
  );
  const acceptanceProbability = mhAcceptanceProbability(logRatio);
  const u = rng.next();
  const accepted = u < acceptanceProbability;
  return { next: accepted ? candidate : current, candidate, accepted, acceptanceProbability };
}

export interface MhChainResult<S> {
  readonly states: readonly S[];
  readonly accepted: readonly boolean[];
  readonly acceptanceRate: number;
}

export function metropolisHastingsChain<S>(
  rng: Rng,
  initial: S,
  nSteps: number,
  sampler: MhSampler<S>,
): MhChainResult<S> {
  const states: S[] = [initial];
  const accepted: boolean[] = [];
  let current = initial;
  for (let i = 0; i < nSteps; i++) {
    const step = metropolisHastingsStep(rng, current, sampler);
    current = step.next;
    states.push(current);
    accepted.push(step.accepted);
  }
  const acceptanceRate = accepted.reduce((s, a) => s + (a ? 1 : 0), 0) / accepted.length;
  return { states, accepted, acceptanceRate };
}

/**
 * An isotropic Gaussian random-walk proposal on `R^D`, PRML Figure 11.9/11.10: the
 * blue circle whose radius is the `stepSize` slider in the chapter's widget.
 * Symmetric, so `proposalLogPdf` is never evaluated where it would matter, but a real
 * isotropic density is still returned rather than a constant, so a caller building a
 * non-symmetric variant by composition sees the correct shape to override.
 */
export function gaussianRandomWalkProposal(
  stepSize: number,
): Pick<MhSampler<number[]>, 'proposalSample' | 'proposalLogPdf'> {
  return {
    proposalSample: (rng, current) => current.map((c) => c + stepSize * standardNormal(rng)),
    proposalLogPdf: (to, from) => {
      let sum = 0;
      for (let i = 0; i < to.length; i++) {
        const d = to[i]! - from[i]!;
        sum += -0.5 * Math.log(2 * Math.PI * stepSize * stepSize) - (d * d) / (2 * stepSize * stepSize);
      }
      return sum;
    },
  };
}

/**
 * The standard sample autocorrelation of a scalar chain, `rho_k = c_k / c_0` for lags
 * `0..maxLag`. Not a book equation: it is what makes the chapter's claim "correlation
 * time replaces rejection rate as the cost" a number instead of an assertion, plotted
 * against the Metropolis-Hastings step-size slider.
 */
export function autocorrelation(chain: readonly number[], maxLag: number): number[] {
  const n = chain.length;
  const mean = chain.reduce((s, x) => s + x, 0) / n;
  const centered = chain.map((x) => x - mean);
  const c0 = centered.reduce((s, x) => s + x * x, 0) / n;
  const out = new Array<number>(maxLag + 1);
  for (let k = 0; k <= maxLag; k++) {
    let sum = 0;
    for (let t = 0; t < n - k; t++) sum += centered[t]! * centered[t + k]!;
    out[k] = c0 === 0 ? (k === 0 ? 1 : 0) : sum / n / c0;
  }
  return out;
}
