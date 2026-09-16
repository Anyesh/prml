import type { Mat, Vec } from '../types.js';
import { inverse } from '../linalg/decompose.js';
import { diag, matmul, matvec, transpose, vecSub } from '../linalg/core.js';

export interface FaParams {
  readonly mean: Vec;
  /** D x M, same layout as `PpcaParams.w`. */
  readonly w: Mat;
  /** Diagonal noise variance per observed dimension, in place of PPCA's single `sigma^2`. */
  readonly psi: Vec;
}

/** PRML's factor-analysis marginal covariance, `W W^T + Psi`: anisotropic noise, unlike PPCA's isotropic `sigma^2 I`. */
export function faMarginalCov(w: Mat, psi: Vec): Mat {
  const wwT = matmul(w, transpose(w));
  return wwT.map((row, i) => row.map((v, j) => v + (i === j ? psi[i]! : 0)));
}

function psiInvDiag(psi: Vec): Mat {
  return diag(psi.map((p) => 1 / p));
}

export interface FaEStepResult {
  readonly ez: Mat;
  readonly ezz: readonly Mat[];
}

/**
 * The factor-analysis E step: `G = (I + W^T Psi^-1 W)^-1`, `E[z_n] = G W^T Psi^-1 (x_n - mean)`,
 * `E[z_n z_n^T] = G + E[z_n] E[z_n]^T`. `G` plays the role PPCA's `sigma^2 M^-1` plays there,
 * but Psi's per-axis noise keeps it from collapsing to a scalar multiple of `M^-1`.
 */
export function faEStep(data: Mat, params: FaParams): FaEStepResult {
  const { mean, w, psi } = params;
  const psiInv = psiInvDiag(psi);
  const wT = transpose(w);
  const inner = matmul(wT, matmul(psiInv, w)).map((row, i) => row.map((v, j) => v + (i === j ? 1 : 0)));
  const g = inverse(inner);
  const gWtPsiInv = matmul(g, matmul(wT, psiInv));

  const ez: number[][] = [];
  const ezz: Mat[] = [];
  for (const x of data) {
    const zn = matvec(gWtPsiInv, vecSub(x, mean));
    ez.push(zn);
    ezz.push(g.map((row, i) => row.map((v, j) => v + zn[i]! * zn[j]!)));
  }
  return { ez, ezz };
}

export interface FaMStepResult {
  readonly w: Mat;
  readonly psi: Vec;
}

/** The factor-analysis M step, given fixed `ez`/`ezz` from `faEStep`. Psi is floored at `1e-8` against a component collapsing onto a single point. */
export function faMStep(data: Mat, mean: Vec, ez: Mat, ezz: readonly Mat[]): FaMStepResult {
  const n = data.length;
  const d = mean.length;
  const m = ez[0]?.length ?? 0;

  const sumXz: number[][] = Array.from({ length: d }, () => new Array(m).fill(0));
  const s: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const ezTCentered: number[][] = Array.from({ length: m }, () => new Array(d).fill(0));

  data.forEach((x, i) => {
    const diff = vecSub(x, mean);
    const zn = ez[i]!;
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < m; b++) sumXz[a]![b]! += diff[a]! * zn[b]!;
      for (let b = 0; b < d; b++) s[a]![b]! += diff[a]! * diff[b]!;
    }
    for (let a = 0; a < m; a++) for (let b = 0; b < d; b++) ezTCentered[a]![b]! += zn[a]! * diff[b]!;
  });

  const sumZz: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (const e of ezz) for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) sumZz[a]![b]! += e[a]![b]!;

  const w = matmul(sumXz, inverse(sumZz));
  const wEzTCentered = matmul(
    w,
    ezTCentered.map((row) => row.map((v) => v / n)),
  );

  const psi = new Array(d).fill(0);
  for (let dd = 0; dd < d; dd++) psi[dd] = Math.max(s[dd]![dd]! / n - wEzTCentered[dd]![dd]!, 1e-8);

  return { w, psi };
}

export interface FaEmStepResult {
  readonly ez: Mat;
  readonly params: FaParams;
}

export function faEmStep(data: Mat, params: FaParams): FaEmStepResult {
  const { ez, ezz } = faEStep(data, params);
  const { w, psi } = faMStep(data, params.mean, ez, ezz);
  return { ez, params: { mean: params.mean, w, psi } };
}

export interface FaEmFitResult {
  readonly paramsHistory: readonly FaParams[];
}

export function faFitEM(data: Mat, initialParams: FaParams, maxIters: number): FaEmFitResult {
  const paramsHistory: FaParams[] = [initialParams];
  let params = initialParams;
  for (let i = 0; i < maxIters; i++) {
    const step = faEmStep(data, params);
    paramsHistory.push(step.params);
    params = step.params;
  }
  return { paramsHistory };
}
