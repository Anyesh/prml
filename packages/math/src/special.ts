const LANCZOS_G = 7;
const LANCZOS_COEF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/**
 * Lanczos approximation. Accurate to ~1e-13 relative for x > 0, which is the budget the
 * 1e-9 golden tolerance leaves once it propagates through a Dirichlet normaliser.
 */
export function logGamma(x: number): number {
  if (x < 0.5) {
    // Reflection formula avoids the pole cluster near non-positive integers.
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const y = x - 1;
  let a = LANCZOS_COEF[0]!;
  const t = y + LANCZOS_G + 0.5;
  for (let i = 1; i < LANCZOS_COEF.length; i++) {
    a += LANCZOS_COEF[i]! / (y + i);
  }
  return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Derivative of `logGamma`. Required by variational Bayes, where it appears in every E[ln .] term. */
export function digamma(x: number): number {
  let z = x;
  let result = 0;
  // Recurrence shifts into the region where the asymptotic tail below is accurate.
  while (z < 6) {
    result -= 1 / z;
    z += 1;
  }
  const f = 1 / (z * z);
  result +=
    Math.log(z) -
    0.5 / z -
    f *
      (1 / 12 -
        f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f * (1 / 132 - f * (691 / 32760 - f / 12))))));
  return result;
}

export function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

/** `log` of the multivariate gamma function of dimension `d`, the Wishart normaliser. */
export function logMultivariateGamma(x: number, d: number): number {
  let sum = (d * (d - 1)) / 4 * Math.log(Math.PI);
  for (let i = 1; i <= d; i++) {
    sum += logGamma(x - (i - 1) / 2);
  }
  return sum;
}

const GAMMA_ITMAX = 500;
const GAMMA_EPS = 3e-16;
const FPMIN = 1e-300;

/** Series expansion of the regularised lower incomplete gamma, valid for x < a + 1. */
function gammaSeries(a: number, x: number): number {
  const gln = logGamma(a);
  let ap = a;
  let sum = 1 / a;
  let del = sum;
  for (let n = 1; n <= GAMMA_ITMAX; n++) {
    ap += 1;
    del *= x / ap;
    sum += del;
    if (Math.abs(del) < Math.abs(sum) * GAMMA_EPS) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - gln);
}

/** Continued fraction for the regularised upper incomplete gamma, valid for x >= a + 1. */
function gammaContinuedFraction(a: number, x: number): number {
  const gln = logGamma(a);
  let b = x + 1 - a;
  let c = 1 / FPMIN;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i <= GAMMA_ITMAX; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < GAMMA_EPS) break;
  }
  return Math.exp(-x + a * Math.log(x) - gln) * h;
}

/** Regularised lower incomplete gamma, `P(a, x)`. Backs the gamma and chi-squared CDFs. */
export function gammaincLower(a: number, x: number): number {
  if (x === 0) return 0;
  // The series converges quickly near the origin; past x = a + 1 the continued fraction
  // for the complementary Q(a, x) does, so P is taken as 1 - Q there instead of forcing
  // the series to converge slowly (or not stably) far past its natural regime.
  if (x < a + 1) return gammaSeries(a, x);
  return 1 - gammaContinuedFraction(a, x);
}

const ERF_A = 0.5;

/** `erf(x) = sign(x) * P(1/2, x^2)`, so it inherits the incomplete-gamma machinery above. */
export function erf(x: number): number {
  if (x === 0) return 0;
  const p = gammaincLower(ERF_A, x * x);
  return x > 0 ? p : -p;
}

/**
 * Must not be computed as `1 - erf(x)`, which loses all significance for x beyond ~3.
 * For x >= 0 with x^2 past the gamma series' natural regime, the continued fraction for
 * Q(1/2, x^2) is evaluated directly instead of subtracting from 1. For x < 0 the identity
 * `erfc(x) = 1 + erf(-x)` is an addition of two positive quantities, never a subtraction.
 */
export function erfc(x: number): number {
  if (x < 0) return 1 + erf(-x);
  const t = x * x;
  if (t < ERF_A + 1) return 1 - gammaSeries(ERF_A, t);
  return gammaContinuedFraction(ERF_A, t);
}

const BETA_ITMAX = 500;
const BETA_EPS = 3e-16;

/** Lentz continued fraction used by `betainc`, per Numerical Recipes `betacf`. */
function betaContinuedFraction(a: number, b: number, x: number): number {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= BETA_ITMAX; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < BETA_EPS) break;
  }
  return h;
}

/** Regularised incomplete beta, `I_x(a, b)`. Backs the beta and Student-t CDFs. */
export function betainc(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  // Swap to the mirrored tail when x sits past the continued fraction's fast-converging side.
  if (x < (a + 1) / (a + b + 2)) return (front * betaContinuedFraction(a, b, x)) / a;
  return 1 - (front * betaContinuedFraction(b, a, 1 - x)) / b;
}

const BESSEL_SERIES_THRESHOLD = 15;
const BESSEL_ITMAX = 400;

function logSumExpPositive(logTerms: number[]): number {
  let max = -Infinity;
  for (const t of logTerms) if (t > max) max = t;
  if (!Number.isFinite(max)) return max;
  let sum = 0;
  for (const t of logTerms) sum += Math.exp(t - max);
  return max + Math.log(sum);
}

/** Direct series in log-space; converges for any x but needs O(x) terms, so it is used only below the asymptotic threshold. */
function besselISeries(nu: number, x: number): number {
  const logHalfX = Math.log(x / 2);
  const logTerms: number[] = [];
  for (let k = 0; k < BESSEL_ITMAX; k++) {
    const logTerm = (2 * k + nu) * logHalfX - logGamma(k + 1) - logGamma(nu + k + 1);
    logTerms.push(logTerm);
    if (k > 0 && logTerm < logTerms[0]! - 40) break;
  }
  return Math.exp(logSumExpPositive(logTerms));
}

/** Debye asymptotic expansion, exponentially scaled so it stays finite for x in the hundreds. */
function besselIAsymptotic(nu: number, x: number): number {
  const mu = 4 * nu * nu;
  let term = 1;
  let sum = 1;
  for (let k = 1; k <= 12; k++) {
    term *= -(mu - (2 * k - 1) * (2 * k - 1)) / (k * 8 * x);
    sum += term;
    if (Math.abs(term) < Math.abs(sum) * 1e-18) break;
  }
  const scaled = sum / Math.sqrt(2 * Math.PI * x);
  return Math.exp(x) * scaled;
}

/** Modified Bessel function of the first kind, order `nu`. The von Mises normaliser. */
export function besselI(nu: number, x: number): number {
  if (x === 0) return nu === 0 ? 1 : 0;
  if (x < BESSEL_SERIES_THRESHOLD) return besselISeries(nu, x);
  return besselIAsymptotic(nu, x);
}
