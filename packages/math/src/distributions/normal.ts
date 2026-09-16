import type { Family, Rng } from '../types.js';
import { erfc } from '../special.js';
import { standardNormal } from '../rng.js';

export interface NormalParams {
  readonly mu: number;
  /** Variance, not standard deviation. PRML parameterises by precision or variance throughout; σ never appears alone. */
  readonly sigma2: number;
}

export function normalLogPdf(x: number, p: NormalParams): number {
  const d = x - p.mu;
  return -0.5 * Math.log(2 * Math.PI * p.sigma2) - (d * d) / (2 * p.sigma2);
}

export function normalPdf(x: number, p: NormalParams): number {
  return Math.exp(normalLogPdf(x, p));
}

export function normalCdf(x: number, p: NormalParams): number {
  const z = (x - p.mu) / Math.sqrt(2 * p.sigma2);
  // erfc is used directly rather than 1 - erf(z), so the far tail keeps its precision.
  return z >= 0 ? 1 - 0.5 * erfc(z) : 0.5 * erfc(-z);
}

const RATIONAL_A = [
  -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1,
  2.506628277459239,
];
const RATIONAL_B = [
  -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1,
];
const RATIONAL_C = [
  -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968,
  2.938163982698783,
];
const RATIONAL_D = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];

/**
 * Acklam's rational approximation to the standard normal quantile (~1.15e-9 relative),
 * refined by two Halley steps against `erf`/`erfc` to reach double precision. There is no
 * closed form, so this is the standard way to invert `Phi`.
 */
function standardNormalQuantile(p: number): number {
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let z: number;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    z =
      (((((RATIONAL_C[0]! * q + RATIONAL_C[1]!) * q + RATIONAL_C[2]!) * q + RATIONAL_C[3]!) * q + RATIONAL_C[4]!) *
        q +
        RATIONAL_C[5]!) /
      ((((RATIONAL_D[0]! * q + RATIONAL_D[1]!) * q + RATIONAL_D[2]!) * q + RATIONAL_D[3]!) * q + 1);
  } else if (p <= pHigh) {
    const q = p - 0.5;
    const r = q * q;
    z =
      ((((((RATIONAL_A[0]! * r + RATIONAL_A[1]!) * r + RATIONAL_A[2]!) * r + RATIONAL_A[3]!) * r + RATIONAL_A[4]!) *
        r +
        RATIONAL_A[5]!) *
        q) /
      (((((RATIONAL_B[0]! * r + RATIONAL_B[1]!) * r + RATIONAL_B[2]!) * r + RATIONAL_B[3]!) * r + RATIONAL_B[4]!) *
        r +
        1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    z =
      -(((((RATIONAL_C[0]! * q + RATIONAL_C[1]!) * q + RATIONAL_C[2]!) * q + RATIONAL_C[3]!) * q + RATIONAL_C[4]!) *
        q +
        RATIONAL_C[5]!) /
      ((((RATIONAL_D[0]! * q + RATIONAL_D[1]!) * q + RATIONAL_D[2]!) * q + RATIONAL_D[3]!) * q + 1);
  }
  // Halley's method on Phi(z) - p, using phi(z) = Phi'(z); two steps take the ~1e-9
  // rational seed to full double precision.
  for (let i = 0; i < 2; i++) {
    const e = 0.5 * erfc(-z / Math.SQRT2) - p;
    const u = e * Math.sqrt(2 * Math.PI) * Math.exp((z * z) / 2);
    z -= u / (1 + (z * u) / 2);
  }
  return z;
}

/** Inverse CDF. Needed for credible-interval ribbons, which are drawn at fixed quantiles. */
export function normalQuantile(q: number, p: NormalParams): number {
  return p.mu + Math.sqrt(p.sigma2) * standardNormalQuantile(q);
}

export function normalSample(rng: Rng, p: NormalParams): number {
  return p.mu + Math.sqrt(p.sigma2) * standardNormal(rng);
}

export const Normal: Family<number, NormalParams> = {
  name: 'normal',
  logPdf: normalLogPdf,
  pdf: normalPdf,
  sample: normalSample,
  mean: (p) => p.mu,
};
