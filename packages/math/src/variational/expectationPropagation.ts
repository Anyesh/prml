import type { Vec } from '../types.js';
import { dot, vecSub } from '../linalg/core.js';

/** An isotropic Gaussian site factor `s * N(theta | mean, variance * I)`, PRML 10.213. */
export interface IsotropicSite {
  readonly logScale: number;
  readonly mean: Vec;
  readonly variance: number;
}

/** The current global approximation `q(theta) = N(theta | mean, variance * I)`, PRML 10.212. */
export interface IsotropicGaussian {
  readonly mean: Vec;
  readonly variance: number;
}

/**
 * PRML 10.214-10.215: removes one site's contribution from the current posterior to form
 * the "leave one out" cavity distribution, the quantity every EP update refines against
 * instead of against `q` itself, so a factor is never fit to information it already
 * contributed.
 */
export function isotropicCavity(q: IsotropicGaussian, site: IsotropicSite): IsotropicGaussian {
  const cavityPrecision = 1 / q.variance - 1 / site.variance;
  const variance = 1 / cavityPrecision;
  const mean = q.mean.map((m, i) => m + variance * (1 / site.variance) * (m - site.mean[i]!));
  return { mean, variance };
}

export interface ClutterModel {
  /** Mixing proportion of the broad clutter component, PRML 10.209. */
  readonly clutterWeight: number;
  /** Variance of the clutter component, `a` in `N(x | 0, a I)`. */
  readonly clutterVariance: number;
}

export interface ClutterMomentMatch {
  readonly normaliser: number;
  readonly posterior: IsotropicGaussian;
  /** Posterior probability that this observation was the signal rather than clutter, PRML 10.219. */
  readonly signalProbability: number;
}

function isotropicLogPdf(x: Vec, mean: Vec, variance: number): number {
  const d = x.length;
  const diff = vecSub(x, mean);
  return -0.5 * (d * Math.log(2 * Math.PI * variance) + dot(diff, diff) / variance);
}

/**
 * PRML 10.216-10.219: moment-matches the true factor `f_n(theta) = p(x_n | theta)` against
 * the cavity `q^{\n}`, by computing the exact mean and variance of the tilted distribution
 * `f_n(theta) q^{\n}(theta) / Z_n` for the clutter problem's two-component likelihood. This
 * is the step that makes EP minimise `KL(p || q)` locally rather than `KL(q || p)`: it
 * matches moments of the true (multimodal, in general) tilted density, not a mode of it.
 */
export function clutterMomentMatch(x: Vec, cavity: IsotropicGaussian, model: ClutterModel): ClutterMomentMatch {
  const d = x.length;
  const { clutterWeight: w, clutterVariance: a } = model;

  const signalDensity = Math.exp(isotropicLogPdf(x, cavity.mean, cavity.variance + 1));
  const clutterDensity = Math.exp(isotropicLogPdf(x, new Array(d).fill(0), a));
  const normaliser = (1 - w) * signalDensity + w * clutterDensity;
  const rho = 1 - (w * clutterDensity) / normaliser;

  const diff = vecSub(x, cavity.mean);
  const shrink = cavity.variance / (cavity.variance + 1);
  const mean = cavity.mean.map((m, i) => m + rho * shrink * diff[i]!);

  const sqDist = dot(diff, diff);
  const variance =
    cavity.variance -
    (rho * cavity.variance * cavity.variance) / (cavity.variance + 1) +
    (rho * (1 - rho) * cavity.variance * cavity.variance * sqDist) / (d * (cavity.variance + 1) ** 2);

  return { normaliser, posterior: { mean, variance }, signalProbability: rho };
}

/**
 * PRML 10.220-10.222: recovers the refined site `f~_n` that would have produced the
 * moment-matched posterior on its own when multiplied into the cavity, so the next sweep
 * has a factor to remove again. `variance` can come out negative when `d` is small and the
 * observation is a near-certain outlier; PRML notes EP itself does not guarantee positive
 * site variances; the caller receives it as-is rather than have it silently clamped.
 */
export function refineIsotropicSite(cavity: IsotropicGaussian, moments: ClutterMomentMatch): IsotropicSite {
  const d = cavity.mean.length;
  const sitePrecision = 1 / moments.posterior.variance - 1 / cavity.variance;
  const variance = 1 / sitePrecision;
  const mean = cavity.mean.map(
    (m, i) => m + (variance + cavity.variance) * (1 / cavity.variance) * (moments.posterior.mean[i]! - m),
  );
  const logNormalDensityAtSite = isotropicLogPdf(mean, cavity.mean, variance + cavity.variance);
  const logScale = Math.log(moments.normaliser) - (d / 2) * Math.log(2 * Math.PI * variance) - logNormalDensityAtSite;
  return { logScale, mean, variance };
}

export interface ClutterEpState {
  readonly posterior: IsotropicGaussian;
  readonly sites: readonly IsotropicSite[];
}

/**
 * One left-to-right sweep of expectation propagation over the whole dataset (PRML's own
 * algorithm box, 10.202-10.208, specialised to the clutter problem): each point's site is
 * removed to form a cavity, refit by moment matching against the true likelihood, and
 * reinserted before the next point's cavity is formed, so every update sees the others'
 * most current contribution.
 */
export function clutterEpSweep(data: readonly Vec[], state: ClutterEpState, model: ClutterModel): ClutterEpState {
  let posterior = state.posterior;
  const sites = state.sites.slice();

  for (let n = 0; n < data.length; n++) {
    const cavity = isotropicCavity(posterior, sites[n]!);
    const moments = clutterMomentMatch(data[n]!, cavity, model);
    const site = refineIsotropicSite(cavity, moments);
    sites[n] = site;
    posterior = moments.posterior;
  }

  return { posterior, sites };
}

/** `clutterEpSweep` repeated `sweeps` times, keeping every intermediate state for a step-through widget. */
export function clutterEpFit(data: readonly Vec[], initial: ClutterEpState, model: ClutterModel, sweeps: number): ClutterEpState[] {
  const history: ClutterEpState[] = [initial];
  let state = initial;
  for (let i = 0; i < sweeps; i++) {
    state = clutterEpSweep(data, state, model);
    history.push(state);
  }
  return history;
}

/** Every site initialised uninformative (infinite variance, scale 1), so the first sweep's cavity is just the prior. */
export function clutterEpInit(prior: IsotropicGaussian, n: number): ClutterEpState {
  const uninformative: IsotropicSite = { logScale: 0, mean: new Array(prior.mean.length).fill(0), variance: 1e12 };
  return { posterior: prior, sites: new Array(n).fill(uninformative) };
}
