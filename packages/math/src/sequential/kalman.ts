import type { Mat, Vec } from '../types.js';
import { eye, matAdd, matmul, matSub, matvec, outer, symmetrise, transpose, vecAdd, vecSub } from '../linalg/core.js';
import { inverse } from '../linalg/decompose.js';
import { mvnLogPdf, mvnPdf } from '../distributions/mvn.js';

/** The linear-Gaussian state space model, PRML 13.75-13.77 and 13.84: `theta = {A, Gamma, C, Sigma, mu0, V0}`. */
export interface LdsParams {
  readonly A: Mat;
  readonly Gamma: Mat;
  readonly C: Mat;
  readonly Sigma: Mat;
  readonly mu0: Vec;
  readonly V0: Mat;
}

export interface KalmanStep {
  /** `mu_n`, the filtered mean `E[zn | x1,...,xn]` (PRML 13.84, 13.89, 13.94). */
  readonly mean: Vec;
  readonly cov: Mat;
  /** `A mu_{n-1}` for n > 0, or `mu0` at n = 0: the state estimate before this step's observation. */
  readonly predictedMean: Vec;
  /** `P_{n-1} = A V_{n-1} A^T + Gamma` for n > 0, or `V0` at n = 0 (PRML 13.88). */
  readonly predictedCov: Mat;
  /** The Kalman gain `K_n` (PRML 13.92, 13.97). */
  readonly gain: Mat;
  /** `c_n = N(xn | C * predictedMean, C * predictedCov * C^T + Sigma)` (PRML 13.91, 13.96). */
  readonly c: number;
  readonly logC: number;
}

/** PRML 13.88 (predicted mean and covariance): project the previous filtered belief forward one transition. */
export function kalmanPredict(mean: Vec, cov: Mat, A: Mat, Gamma: Mat): { mean: Vec; cov: Mat } {
  return {
    mean: matvec(A, mean),
    cov: symmetrise(matAdd(matmul(matmul(A, cov), transpose(A)), Gamma)),
  };
}

/**
 * PRML 13.89-13.92 (also 13.94-13.97 at n = 1, which is this same update against the prior
 * `p(z1)` rather than a transitioned belief): correct a predicted state with one
 * observation, producing the filtered mean/covariance, the gain, and the innovation
 * density `c_n` used both for the likelihood (13.63) and, unscaled, for nothing else -
 * unlike the HMM there is no analogous underflow risk here because each `c_n` is a
 * bounded Gaussian density rather than a product accumulating across the whole chain.
 */
export function kalmanUpdate(
  predictedMean: Vec,
  predictedCov: Mat,
  x: Vec,
  C: Mat,
  Sigma: Mat,
): { mean: Vec; cov: Mat; gain: Mat; c: number; logC: number } {
  const dim = predictedMean.length;
  const ct = transpose(C);
  // C P C^T is symmetric mathematically for any symmetric P, but matmul's rounding does
  // not preserve that exactly, and this feeds both the gain and, below, mvnPdf's own
  // Cholesky factorisation, which rejects a merely-close-to-symmetric matrix outright.
  const innovationCov = symmetrise(matAdd(matmul(matmul(C, predictedCov), ct), Sigma));
  const innovationCovInv = inverse(innovationCov);
  const gain = matmul(matmul(predictedCov, ct), innovationCovInv);
  const predictedObs = matvec(C, predictedMean);
  const residual = vecSub(x, predictedObs);
  const mean = vecAdd(predictedMean, matvec(gain, residual));
  // The textbook (I - KC) P form is symmetric only in exact arithmetic; floating point
  // drift compounds across a long filter run or many EM iterations until Cholesky
  // (inside the next step's mvnPdf) rejects it as "not positive definite" on a matrix
  // that is symmetric up to noise but not exactly. Symmetrising here is what
  // `gmm.ts`'s M-step already does for exactly this reason.
  const cov = symmetrise(matmul(matSub(eye(dim), matmul(gain, C)), predictedCov));
  const mvn = { mean: predictedObs, cov: innovationCov };
  return { mean, cov, gain, c: mvnPdf(x, mvn), logC: mvnLogPdf(x, mvn) };
}

/**
 * The Kalman filter (PRML 13.3.1): `kalmanPredict` then `kalmanUpdate` at every step,
 * except the first, which corrects the prior `p(z1) = N(mu0, V0)` directly (13.93-13.97)
 * rather than a transitioned belief, because `z1` has no predecessor to transition from.
 */
export function kalmanFilter(observations: Mat, params: LdsParams): KalmanStep[] {
  const steps: KalmanStep[] = [];
  let mean = params.mu0;
  let cov = params.V0;
  observations.forEach((x, n) => {
    const predicted = n === 0 ? { mean: params.mu0, cov: params.V0 } : kalmanPredict(mean, cov, params.A, params.Gamma);
    const updated = kalmanUpdate(predicted.mean, predicted.cov, x, params.C, params.Sigma);
    steps.push({
      mean: updated.mean,
      cov: updated.cov,
      predictedMean: predicted.mean,
      predictedCov: predicted.cov,
      gain: updated.gain,
      c: updated.c,
      logC: updated.logC,
    });
    mean = updated.mean;
    cov = updated.cov;
  });
  return steps;
}

/** PRML 13.63 read off in log space, the numerically safe way to accumulate the LDS's own analogue of the HMM's scaling factors. */
export function kalmanLogLikelihood(logC: Vec): number {
  return logC.reduce((a, b) => a + b, 0);
}

export interface KalmanSmoothResult {
  /** `mu_hat_n`, PRML 13.100. Index `N-1` equals the filter's own `mean[N-1]`: there is nothing later to smooth against. */
  readonly mean: Vec[];
  readonly cov: Mat[];
  /** `J_n` (PRML 13.102), one per transition, `length = steps.length - 1`. */
  readonly J: Mat[];
  /** `cov[z_{n+1}, z_n]` (PRML 13.104), indexed by the earlier time `n`, `length = steps.length - 1`. */
  readonly pairwiseCov: Mat[];
}

/**
 * The RTS smoother (PRML 13.3.1, 13.100-13.104): a backward pass over an already-completed
 * `kalmanFilter` run, using `beta` in its "alpha-gamma" LDS form rather than the HMM's own
 * "alpha-beta" form, exactly the choice the book calls out (end of 13.2.4) as the one that
 * differs between the two chains sharing this same message-passing structure.
 */
export function kalmanSmoother(steps: readonly KalmanStep[], params: LdsParams): KalmanSmoothResult {
  const n = steps.length;
  const mean: Vec[] = new Array(n);
  const cov: Mat[] = new Array(n);
  const J: Mat[] = new Array(n - 1);
  const pairwiseCov: Mat[] = new Array(n - 1);

  mean[n - 1] = steps[n - 1]!.mean;
  cov[n - 1] = steps[n - 1]!.cov;

  for (let t = n - 2; t >= 0; t--) {
    const step = steps[t]!;
    const next = steps[t + 1]!;
    const jt = matmul(matmul(step.cov, transpose(params.A)), inverse(next.predictedCov));
    J[t] = jt;
    mean[t] = vecAdd(step.mean, matvec(jt, vecSub(mean[t + 1]!, next.predictedMean)));
    const covGap = matSub(cov[t + 1]!, next.predictedCov);
    cov[t] = symmetrise(matAdd(step.cov, matmul(matmul(jt, covGap), transpose(jt))));
    pairwiseCov[t] = matmul(jt, cov[t + 1]!);
  }

  return { mean, cov, J, pairwiseCov };
}

export interface KalmanExpectations {
  readonly Ez: Vec[];
  readonly Ezz: Mat[];
  /** `E[z(t+1) z(t)^T]`, `t = 0, ..., N-2` (PRML 13.106, via `cov[zn,zn-1]` of 13.104). */
  readonly Ezzlag: Mat[];
}

/** PRML 13.105-13.107: the sufficient statistics the M step needs, built from a completed `kalmanSmoother` pass. */
export function kalmanEmExpectations(smoothed: KalmanSmoothResult): KalmanExpectations {
  const Ez = smoothed.mean;
  const Ezz = smoothed.cov.map((v, n) => matAdd(v, outer(Ez[n]!, Ez[n]!)));
  const Ezzlag = smoothed.pairwiseCov.map((p, t) => matAdd(p, outer(Ez[t + 1]!, Ez[t]!)));
  return { Ez, Ezz, Ezzlag };
}

function sumMats(mats: readonly Mat[]): Mat {
  return mats.reduce((acc, m) => matAdd(acc, m));
}

/**
 * The M step of LDS-EM (PRML 13.3.2, 13.110-13.116): `mu0`/`V0` from the first time
 * slice's moments, `A`/`Gamma` from the lagged second moments across every transition,
 * `C`/`Sigma` from the observations against the smoothed state. `A` must be computed
 * before `Gamma`, and `C` before `Sigma`, because each pair's second update substitutes
 * the first's result (the book's own note after 13.114).
 */
export function kalmanMStep(observations: Mat, expectations: KalmanExpectations): LdsParams {
  const { Ez, Ezz, Ezzlag } = expectations;
  const n = observations.length;

  const mu0 = Ez[0]!;
  const V0 = symmetrise(matSub(Ezz[0]!, outer(Ez[0]!, Ez[0]!)));

  const sumEzzlag = sumMats(Ezzlag);
  const sumEzzPrev = sumMats(Ezz.slice(0, n - 1));
  const A = matmul(sumEzzlag, inverse(sumEzzPrev));
  const At = transpose(A);

  const gammaTerms: Mat[] = [];
  for (let t = 0; t < n - 1; t++) {
    const nextEzz = Ezz[t + 1]!;
    const lag = Ezzlag[t]!;
    const term = matAdd(
      matSub(nextEzz, matmul(A, transpose(lag))),
      matSub(matmul(A, matmul(Ezz[t]!, At)), matmul(lag, At)),
    );
    gammaTerms.push(term);
  }
  const Gamma = symmetrise(sumMats(gammaTerms).map((row) => row.map((v) => v / (n - 1))));

  const sumXEz = sumMats(observations.map((x, t) => outer(x, Ez[t]!)));
  const sumEzzAll = sumMats(Ezz);
  const C = matmul(sumXEz, inverse(sumEzzAll));
  const Ct = transpose(C);

  const sigmaTerms: Mat[] = observations.map((x, t) => {
    const xxT = outer(x, x);
    const cEzxT = matmul(C, outer(Ez[t]!, x));
    const xEzTCt = matmul(outer(x, Ez[t]!), Ct);
    const cEzzCt = matmul(matmul(C, Ezz[t]!), Ct);
    return matAdd(matSub(xxT, matAdd(cEzxT, xEzTCt)), cEzzCt);
  });
  const Sigma = symmetrise(sumMats(sigmaTerms).map((row) => row.map((v) => v / n)));

  return { A, Gamma, C, Sigma, mu0, V0 };
}
