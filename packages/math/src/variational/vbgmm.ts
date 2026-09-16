import type { Mat, Vec } from '../types.js';
import { logGamma, logMultivariateGamma } from '../special.js';
import { softmax } from '../numeric.js';
import { logDet, inverse } from '../linalg/decompose.js';
import { outer, quadForm, symmetrise, trace } from '../linalg/core.js';
import { wishartExpectedLogDet, type WishartParams } from '../distributions/wishart.js';
import { dirichletExpectedLog, type DirichletParams } from '../distributions/dirichlet.js';
import { multivariateTLogPdf } from '../distributions/studentT.js';

/**
 * The Gaussian-Wishart posterior over one component's `(mean, precision)`, PRML 10.59.
 * `scale`/`dof` are the Wishart's own `W`/`nu`, reused verbatim rather than renamed, so a
 * caller can hand `{ scale, dof }` straight to `wishartExpectedLogDet`.
 */
export interface GaussianWishart {
  readonly beta: number;
  readonly mean: Vec;
  readonly scale: Mat;
  readonly dof: number;
}

export interface VbGmmPosterior {
  /** `q(pi) = Dir(pi | alpha)`, one entry per component. */
  readonly alpha: Vec;
  readonly components: readonly GaussianWishart[];
}

/** The symmetric conjugate prior shared by every component, PRML 10.39-10.40. */
export interface VbGmmPrior {
  readonly alpha0: number;
  readonly beta0: number;
  readonly mean0: Vec;
  readonly scale0: Mat;
  readonly dof0: number;
}

/**
 * `-ln B(W, nu)`, the Wishart normalising constant used by PRML 10.74. Not exported from
 * `distributions/wishart.ts` (outside this chapter's owned paths this wave), so it is
 * reproduced here from the same two primitives that file itself uses; see the chapter
 * report for the follow-up to export it centrally instead.
 */
function wishartLogNormaliser(scale: Mat, dof: number): number {
  const d = scale.length;
  return -(dof / 2) * logDet(scale) - ((dof * d) / 2) * Math.log(2) - logMultivariateGamma(dof / 2, d);
}

/** `ln C(alpha) = ln Gamma(sum alpha) - sum ln Gamma(alpha_k)`, the Dirichlet normaliser used by 10.73 and 10.76. */
function dirichletLogNormaliser(alpha: Vec): number {
  let sumAlpha = 0;
  let sumLogGamma = 0;
  for (const a of alpha) {
    sumAlpha += a;
    sumLogGamma += logGamma(a);
  }
  return logGamma(sumAlpha) - sumLogGamma;
}

/** `E[ln pi_k]`, PRML 10.66, one entry per component. */
export function vbGmmExpectedLogPi(posterior: VbGmmPosterior): number[] {
  return dirichletExpectedLog({ alpha: posterior.alpha } satisfies DirichletParams);
}

/** `E[ln |Lambda_k|]`, PRML 10.65, one entry per component. */
export function vbGmmExpectedLogDet(posterior: VbGmmPosterior): number[] {
  return posterior.components.map((c) =>
    wishartExpectedLogDet({ scale: symmetrise(c.scale), nu: c.dof } satisfies WishartParams),
  );
}

/**
 * PRML 10.46, 10.64: the quadratic form `E[(x - mu_k)^T Lambda_k (x - mu_k)]` under the
 * Gaussian-Wishart posterior, which is `D/beta_k` plus a Mahalanobis distance through the
 * Wishart's expected precision `nu_k W_k`, not through `W_k` alone.
 */
export function vbGmmExpectedQuadratic(x: Vec, component: GaussianWishart): number {
  const d = x.length;
  const diff = x.map((v, i) => v - component.mean[i]!);
  return d / component.beta + component.dof * quadForm(diff, component.scale, diff);
}

/**
 * The variational E-step, PRML 10.46-10.49 and 10.67 combined: log responsibilities are
 * assembled from the expected log mixing weight, the expected log precision determinant,
 * and the expected quadratic form, then normalised per point with `softmax` so the
 * log-domain sum never needs its own denominator pass.
 */
export function vbGmmResponsibilities(data: Mat, posterior: VbGmmPosterior): Mat {
  const d = data[0]?.length ?? 0;
  const expectedLogPi = vbGmmExpectedLogPi(posterior);
  const expectedLogDet = vbGmmExpectedLogDet(posterior);
  return data.map((x) => {
    const logRho = posterior.components.map((c, k) => {
      const quad = vbGmmExpectedQuadratic(x, c);
      return expectedLogPi[k]! + 0.5 * expectedLogDet[k]! - (d / 2) * Math.log(2 * Math.PI) - 0.5 * quad;
    });
    return softmax(logRho);
  });
}

interface WeightedStatistics {
  readonly nk: number[];
  readonly xbar: Vec[];
  readonly sk: Mat[];
}

/** PRML 10.51-10.53: the responsibility-weighted count, mean, and covariance per component. */
function weightedStatistics(data: Mat, r: Mat): WeightedStatistics {
  const n = data.length;
  const k = r[0]?.length ?? 0;
  const d = data[0]?.length ?? 0;

  const nk = new Array(k).fill(0);
  for (const row of r) for (let c = 0; c < k; c++) nk[c]! += row[c]!;

  const xbar: number[][] = Array.from({ length: k }, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      const w = r[i]![c]!;
      for (let dim = 0; dim < d; dim++) xbar[c]![dim]! += w * data[i]![dim]!;
    }
  }
  for (let c = 0; c < k; c++) if (nk[c]! > 0) for (let dim = 0; dim < d; dim++) xbar[c]![dim]! /= nk[c]!;

  const sk: number[][][] = Array.from({ length: k }, () => Array.from({ length: d }, () => new Array(d).fill(0)));
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      const w = r[i]![c]!;
      const diff = data[i]!.map((v, dim) => v - xbar[c]![dim]!);
      const contribution = outer(diff, diff);
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) sk[c]![a]![b]! += w * contribution[a]![b]!;
    }
  }
  for (let c = 0; c < k; c++) if (nk[c]! > 0) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) sk[c]![a]![b]! /= nk[c]!;

  return { nk, xbar, sk: sk.map(symmetrise) };
}

/**
 * The variational M-step, PRML 10.58 and 10.60-10.63: the Dirichlet and Gaussian-Wishart
 * posteriors given fixed responsibilities. A component with `Nk` at or near zero reverts
 * every one of its parameters to the prior exactly, which is the mechanism behind
 * automatic pruning: nothing special-cases a "dead" component, the same update just has
 * no data left to move it.
 */
export function vbGmmMStep(data: Mat, r: Mat, prior: VbGmmPrior): VbGmmPosterior {
  const { nk, xbar, sk } = weightedStatistics(data, r);
  const k = nk.length;
  const d = prior.mean0.length;

  const alpha = nk.map((n) => prior.alpha0 + n);

  const components: GaussianWishart[] = [];
  for (let c = 0; c < k; c++) {
    const beta = prior.beta0 + nk[c]!;
    const mean = prior.mean0.map((m0, dim) => (prior.beta0 * m0 + nk[c]! * xbar[c]![dim]!) / beta);

    const scale0Inv = inverse(prior.scale0);
    const meanDiff = xbar[c]!.map((v, dim) => v - prior.mean0[dim]!);
    const meanOuter = outer(meanDiff, meanDiff);
    const shrinkage = (prior.beta0 * nk[c]!) / (prior.beta0 + nk[c]!);

    const scaleInv: number[][] = Array.from({ length: d }, (_, a) =>
      Array.from({ length: d }, (_, b) => scale0Inv[a]![b]! + nk[c]! * sk[c]![a]![b]! + shrinkage * meanOuter[a]![b]!),
    );
    const scale = symmetrise(inverse(symmetrise(scaleInv)));
    const dof = prior.dof0 + nk[c]!;

    components.push({ beta, mean, scale, dof });
  }

  return { alpha, components };
}

/** One responsibility computation followed by one parameter update; the variational analogue of `gmmEmStep`. */
export function vbGmmStep(data: Mat, posterior: VbGmmPosterior, prior: VbGmmPrior): { responsibilities: Mat; posterior: VbGmmPosterior } {
  const responsibilities = vbGmmResponsibilities(data, posterior);
  return { responsibilities, posterior: vbGmmMStep(data, responsibilities, prior) };
}

export interface VbGmmFitResult {
  readonly posteriorHistory: readonly VbGmmPosterior[];
  readonly responsibilitiesHistory: readonly Mat[];
}

/** `vbGmmStep` iterated `iterations` times from `initial`, keeping every intermediate posterior for a step-through widget. */
export function vbGmmFit(data: Mat, initial: VbGmmPosterior, prior: VbGmmPrior, iterations: number): VbGmmFitResult {
  const posteriorHistory: VbGmmPosterior[] = [initial];
  const responsibilitiesHistory: Mat[] = [];
  let posterior = initial;
  for (let i = 0; i < iterations; i++) {
    const step = vbGmmStep(data, posterior, prior);
    responsibilitiesHistory.push(step.responsibilities);
    posteriorHistory.push(step.posterior);
    posterior = step.posterior;
  }
  return { posteriorHistory, responsibilitiesHistory };
}

function matmulSquare(a: Mat, b: Mat): number[][] {
  const n = a.length;
  const out: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += a[i]![k]! * b[k]![j]!;
      out[i]![j] = sum;
    }
  }
  return out;
}

/** PRML 10.71: the responsibility-weighted expected complete-data log-likelihood term of the lower bound. */
function expectedLogPX(data: Mat, posterior: VbGmmPosterior, nk: number[], xbar: Vec[], sk: Mat[]): number {
  const d = data[0]?.length ?? 0;
  const expectedLogDet = vbGmmExpectedLogDet(posterior);
  let total = 0;
  posterior.components.forEach((c, k) => {
    if (nk[k]! === 0) return;
    const diff = xbar[k]!.map((v, dim) => v - c.mean[dim]!);
    const mahalanobis = quadForm(diff, c.scale, diff);
    total +=
      nk[k]! *
      (expectedLogDet[k]! - d / c.beta - c.dof * trace(matmulSquare(sk[k]!, c.scale)) - c.dof * mahalanobis - d * Math.log(2 * Math.PI));
  });
  return 0.5 * total;
}

/**
 * PRML 10.70-10.77: the variational lower bound for the Gaussian mixture, as the sum of
 * the four expected joint-density terms minus the three entropy terms. Each term is kept
 * as its own local computation, named after its equation, rather than folded into one
 * expression, because that is what makes 10.70's seven-term decomposition checkable term
 * by term against the golden fixture.
 */
export function vbGmmLowerBound(data: Mat, r: Mat, posterior: VbGmmPosterior, prior: VbGmmPrior): number {
  const { nk, xbar, sk } = weightedStatistics(data, r);
  const d = prior.mean0.length;
  const k = posterior.components.length;
  const expectedLogPi = vbGmmExpectedLogPi(posterior);
  const expectedLogDet = vbGmmExpectedLogDet(posterior);

  const eLnPX = expectedLogPX(data, posterior, nk, xbar, sk);

  let eLnPZ = 0;
  for (let n = 0; n < data.length; n++) for (let c = 0; c < k; c++) eLnPZ += r[n]![c]! * expectedLogPi[c]!;

  const alpha0Vec = new Array(k).fill(prior.alpha0);
  const eLnPPi = dirichletLogNormaliser(alpha0Vec) + (prior.alpha0 - 1) * expectedLogPi.reduce((s, v) => s + v, 0);

  const scale0Inv = inverse(prior.scale0);
  let eLnPMuLambda = 0;
  for (let c = 0; c < k; c++) {
    const comp = posterior.components[c]!;
    const diff = comp.mean.map((v, dim) => v - prior.mean0[dim]!);
    const mahalanobis = quadForm(diff, comp.scale, diff);
    eLnPMuLambda +=
      0.5 *
      (d * Math.log(prior.beta0 / (2 * Math.PI)) + expectedLogDet[c]! - (d * prior.beta0) / comp.beta - prior.beta0 * comp.dof * mahalanobis);
    eLnPMuLambda += ((prior.dof0 - d - 1) / 2) * expectedLogDet[c]!;
    eLnPMuLambda -= 0.5 * comp.dof * trace(matmulSquare(scale0Inv, comp.scale));
  }
  eLnPMuLambda += k * wishartLogNormaliser(prior.scale0, prior.dof0);

  let eLnQZ = 0;
  for (const row of r) for (const v of row) if (v > 0) eLnQZ += v * Math.log(v);

  const eLnQPi = posterior.alpha.reduce((s, a, c) => s + (a - 1) * expectedLogPi[c]!, 0) + dirichletLogNormaliser(posterior.alpha);

  let eLnQMuLambda = 0;
  for (let c = 0; c < k; c++) {
    const comp = posterior.components[c]!;
    const entropyWishart =
      -wishartLogNormaliser(comp.scale, comp.dof) - ((comp.dof - d - 1) / 2) * expectedLogDet[c]! + (comp.dof * d) / 2;
    eLnQMuLambda += 0.5 * expectedLogDet[c]! + (d / 2) * Math.log(comp.beta / (2 * Math.PI)) - d / 2 - entropyWishart;
  }

  return eLnPX + eLnPZ + eLnPPi + eLnPMuLambda - eLnQZ - eLnQPi - eLnQMuLambda;
}

/**
 * PRML 10.81-10.82: the predictive density is a mixture of Student-t distributions, one
 * per component, weighted by the posterior's own expected mixing coefficients. `L_k`
 * plays the role of a precision in `multivariateTLogPdf`, which parameterises by the
 * matching covariance-like scale, so it is inverted once per component per call.
 */
export function vbGmmPredictiveLogPdf(x: Vec, posterior: VbGmmPosterior): number {
  const d = x.length;
  const alphaHat = posterior.alpha.reduce((s, a) => s + a, 0);
  const terms = posterior.components.map((c, k) => {
    const nu = c.dof + 1 - d;
    const precisionScale = ((c.dof + 1 - d) * c.beta) / (1 + c.beta);
    const lk = c.scale.map((row) => row.map((v) => v * precisionScale));
    const scale = symmetrise(inverse(lk));
    const logPdf = multivariateTLogPdf(x, { mean: c.mean, scale, nu });
    return { weight: posterior.alpha[k]! / alphaHat, logPdf };
  });
  const max = Math.max(...terms.map((t) => t.logPdf));
  const sum = terms.reduce((s, t) => s + t.weight * Math.exp(t.logPdf - max), 0);
  return max + Math.log(sum);
}

/** Prior from which every component starts identical, so responsibilities alone break the symmetry at the first E-step. */
export function vbGmmInitFromPrior(prior: VbGmmPrior, k: number): VbGmmPosterior {
  return {
    alpha: new Array(k).fill(prior.alpha0),
    components: Array.from({ length: k }, () => ({
      beta: prior.beta0,
      mean: [...prior.mean0],
      scale: prior.scale0.map((row) => [...row]),
      dof: prior.dof0,
    })),
  };
}
