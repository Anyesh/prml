import { logSumExp, softmax } from '../numeric.js';
import type { Vec } from '../types.js';

export interface MdnParams {
  readonly mixing: readonly number[];
  readonly sigma: readonly number[];
  /** `means[k]` is component `k`'s `L`-dimensional mean. */
  readonly means: readonly (readonly number[])[];
}

/**
 * PRML 5.150-5.152: a single linear-output network emits `K*(2+L)` raw activations,
 * laid out as mixing logits, then log-variances, then means, and this is the only
 * place that ordering is decided, so `mixing`/`sigma`/`means` elsewhere never see the
 * raw layout again.
 */
export function mdnParamsFromOutput(raw: Vec, numComponents: number, targetDim: number): MdnParams {
  const rawPi = raw.slice(0, numComponents);
  const rawSigma = raw.slice(numComponents, 2 * numComponents);
  const rawMu = raw.slice(2 * numComponents, 2 * numComponents + numComponents * targetDim);
  const mixing = softmax(rawPi);
  const sigma = rawSigma.map(Math.exp);
  const means: number[][] = [];
  for (let k = 0; k < numComponents; k++) means.push(rawMu.slice(k * targetDim, (k + 1) * targetDim));
  return { mixing, sigma, means };
}

function squaredDistance(a: Vec, b: Vec): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i]! - b[i]!) * (a[i]! - b[i]!);
  return sum;
}

/** Isotropic Gaussian log-density for one component, the `N(t | μ_k, σ_k²)` factor inside 5.148. */
export function mdnComponentLogPdf(target: Vec, mean: Vec, sigma: number): number {
  const l = target.length;
  return -0.5 * l * Math.log(2 * Math.PI) - l * Math.log(sigma) - squaredDistance(target, mean) / (2 * sigma * sigma);
}

/** PRML 5.153: `-ln Σ_k π_k N(t | μ_k, σ_k²)`, via `logSumExp` for the same underflow reason every mixture log-likelihood in this package uses it. */
export function mdnErrorSingle(params: MdnParams, target: Vec): number {
  const logTerms = params.mixing.map((pi, k) => Math.log(pi) + mdnComponentLogPdf(target, params.means[k]!, params.sigma[k]!));
  return -logSumExp(logTerms);
}

/** PRML 5.154: the posterior responsibility of component `k` for this target, given the mixture it came from. */
export function mdnResponsibilities(params: MdnParams, target: Vec): number[] {
  const logTerms = params.mixing.map((pi, k) => Math.log(pi) + mdnComponentLogPdf(target, params.means[k]!, params.sigma[k]!));
  const lse = logSumExp(logTerms);
  return logTerms.map((lt) => Math.exp(lt - lse));
}

export interface MdnOutputGradients {
  readonly dMixing: readonly number[];
  readonly dSigma: readonly number[];
  readonly dMeans: readonly (readonly number[])[];
}

/**
 * PRML 5.155-5.157: the error derivatives with respect to the network's raw
 * activations, not the transformed mixing/sigma/mean values, so backpropagation can
 * treat this exactly like any other canonical output delta and hand it straight to
 * `propagateDeltas`.
 */
export function mdnOutputGradients(params: MdnParams, target: Vec): MdnOutputGradients {
  const gamma = mdnResponsibilities(params, target);
  const l = target.length;
  const dMixing = params.mixing.map((pi, k) => pi - gamma[k]!);
  const dSigma = params.sigma.map((sigma, k) => {
    const d2 = squaredDistance(target, params.means[k]!);
    return -gamma[k]! * (d2 / (sigma * sigma) - l);
  });
  const dMeans = params.means.map((mean, k) => mean.map((mu, j) => (gamma[k]! * (mu - target[j]!)) / (params.sigma[k]! * params.sigma[k]!)));
  return { dMixing, dSigma, dMeans };
}

/** PRML 5.158: the mixture's conditional mean, `Σ_k π_k(x) μ_k(x)`. */
export function mdnPredictiveMean(params: MdnParams): number[] {
  const l = params.means[0]!.length;
  const mean = new Array<number>(l).fill(0);
  params.mixing.forEach((pi, k) => {
    params.means[k]!.forEach((mu, j) => {
      mean[j] = mean[j]! + pi * mu;
    });
  });
  return mean;
}

/** PRML 5.159-5.160: total predictive variance around the single mean above, which multimodal targets make a poor summary of the density. */
export function mdnPredictiveVariance(params: MdnParams): number {
  const mean = mdnPredictiveMean(params);
  const l = mean.length;
  let variance = 0;
  params.mixing.forEach((pi, k) => {
    variance += pi * (l * params.sigma[k]! * params.sigma[k]! + squaredDistance(params.means[k]!, mean));
  });
  return variance;
}
