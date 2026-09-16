import { mvnSample, pcg32, polynomialBasis, sigmoid, standardNormal } from '@prml/math';

const BMA_SEED = 20260920;
const COMMITTEE_SEED = 20260921;
const BOOST_SEED = 20260922;
const TREE_REGRESSION_SEED = 20260923;
const TREE_CLASSIFICATION_SEED = 20260924;
const MIXTURE_REGRESSION_SEED = 20260925;
const MIXTURE_LOGISTIC_SEED = 20260926;
const EXPERTS_SEED = 20260927;

/**
 * The two candidate one-dimensional Gaussian "models" for 14.1's Bayesian-model-averaging
 * panel, and the two mixture components sharing the same means/variance for its
 * model-combination panel. Deliberately the same shapes in both panels, so the only thing
 * that differs between the two panels is which generative story produced the data, not
 * how spread out the candidates are.
 */
export const BMA_MODELS = [
  { mu: -1.2, sigma2: 1 },
  { mu: 1.2, sigma2: 1 },
] as const;

/** N points drawn entirely from `BMA_MODELS[trueModel]`, for the "one model generated everything" panel. */
export function bmaSingleModelData(n: number, trueModel: 0 | 1): number[] {
  const rng = pcg32(BMA_SEED, 1);
  const model = BMA_MODELS[trueModel];
  return Array.from({ length: n }, () => model.mu + Math.sqrt(model.sigma2) * standardNormal(rng));
}

/** N points where every single point independently picks one of the two components, for the mixture panel. */
export function bmaMixtureData(n: number, mixing = 0.5): { readonly x: number; readonly source: 0 | 1 }[] {
  const rng = pcg32(BMA_SEED, 2);
  return Array.from({ length: n }, () => {
    const source: 0 | 1 = rng.next() < mixing ? 0 : 1;
    const model = BMA_MODELS[source];
    return { x: model.mu + Math.sqrt(model.sigma2) * standardNormal(rng), source };
  });
}

/**
 * The book's own running sinusoidal regression example (PRML figures 3.5/14.2), used here
 * for the committee widget: `N` points of `sin(2*pi*x)` plus Gaussian noise on `[0, 1]`.
 */
export function committeeSinusoidData(n = 25, noiseSd = 0.25): { readonly x: number[]; readonly t: number[] } {
  const rng = pcg32(COMMITTEE_SEED, 1);
  const x = Array.from({ length: n }, () => rng.next()).sort((a, b) => a - b);
  const t = x.map((xi) => Math.sin(2 * Math.PI * xi) + noiseSd * standardNormal(rng));
  return { x, t };
}

/** Indices of one bootstrap resample of `n` points (sampling with replacement). */
export function bootstrapIndices(rng: ReturnType<typeof pcg32>, n: number): number[] {
  return Array.from({ length: n }, () => Math.floor(rng.next() * n));
}

export const COMMITTEE_DEGREE = 6;
export const COMMITTEE_BASIS = polynomialBasis(COMMITTEE_DEGREE);
export const COMMITTEE_LAMBDA = 1e-3;

/** A fresh RNG stream per committee member, so member `m` is reproducible independent of how many members were drawn before it. */
export function committeeMemberRng(member: number) {
  return pcg32(COMMITTEE_SEED, 100 + member);
}

/**
 * A two-class, two-dimensional toy set in the style of PRML figure A.7 / the AdaBoost
 * illustration of figure 14.2: two overlapping Gaussian blobs, close enough that no single
 * axis-aligned threshold separates them, which is what makes boosting multiple stumps do
 * better than any one of them.
 */
export function boostingToyData(n = 30): { readonly X: number[][]; readonly t: number[] } {
  const rng = pcg32(BOOST_SEED, 1);
  const perClass = Math.floor(n / 2);
  const X: number[][] = [];
  const t: number[] = [];
  const blobs: readonly { center: readonly [number, number]; label: 1 | -1 }[] = [
    { center: [-0.6, -0.5], label: -1 },
    { center: [0.7, 0.6], label: 1 },
  ];
  const cov = [
    [0.45, 0.15],
    [0.15, 0.45],
  ];
  for (const blob of blobs) {
    for (let i = 0; i < perClass; i++) {
      X.push(mvnSample(rng, { mean: [...blob.center], cov }));
      t.push(blob.label);
    }
  }
  return { X, t };
}

/**
 * A 2D regression surface with two genuinely axis-aligned regions of different levels, so
 * a shallow CART regression tree recovers it almost exactly: this is the case a tree is
 * good at, which is the point of using it as this section's worked example.
 */
export function treeRegressionData(n = 60): { readonly X: number[][]; readonly t: number[] } {
  const rng = pcg32(TREE_REGRESSION_SEED, 1);
  const X: number[][] = [];
  const t: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = rng.next() * 4 - 2;
    const x2 = rng.next() * 4 - 2;
    let level: number;
    if (x1 <= 0) level = x2 <= 0.5 ? -1.5 : 0.5;
    else level = x2 <= -0.5 ? 1.5 : 2.5;
    X.push([x1, x2]);
    t.push(level + 0.2 * standardNormal(rng));
  }
  return { X, t };
}

/** A 3-class 2D set with class regions separated well enough for a shallow classification tree. */
export function treeClassificationData(n = 60): { readonly X: number[][]; readonly labels: number[] } {
  const rng = pcg32(TREE_CLASSIFICATION_SEED, 1);
  const centers: readonly (readonly [number, number])[] = [
    [-1.3, -1.3],
    [1.3, -1.3],
    [0, 1.3],
  ];
  const cov = [
    [0.25, 0],
    [0, 0.25],
  ];
  const X: number[][] = [];
  const labels: number[] = [];
  const perClass = Math.floor(n / 3);
  centers.forEach((center, k) => {
    for (let i = 0; i < perClass; i++) {
      X.push(mvnSample(rng, { mean: [...center], cov }));
      labels.push(k);
    }
  });
  return { X, labels };
}

/**
 * PRML figure 14.8's "two crossing lines" toy set for the mixture of linear regression
 * models: `x` uniform on `[-1, 1]`, each point independently assigned to one of two lines.
 */
export function mixtureRegressionData(n = 50): { readonly design: number[][]; readonly t: number[] } {
  const rng = pcg32(MIXTURE_REGRESSION_SEED, 1);
  const design: number[][] = [];
  const t: number[] = [];
  const noiseSd = 0.15;
  for (let i = 0; i < n; i++) {
    const x = rng.next() * 2 - 1;
    const line: 0 | 1 = rng.next() < 0.5 ? 0 : 1;
    const mean = line === 0 ? 0.6 * x + 0.5 : -0.7 * x - 0.4;
    design.push([1, x]);
    t.push(mean + noiseSd * standardNormal(rng));
  }
  return { design, t };
}

/** PRML figure 14.10's style of two-class set that one logistic regression cannot separate but two can. */
export function mixtureLogisticData(n = 60): { readonly design: number[][]; readonly targets: number[] } {
  const rng = pcg32(MIXTURE_LOGISTIC_SEED, 1);
  const design: number[][] = [];
  const targets: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = rng.next() * 4 - 2;
    const x2 = rng.next() * 4 - 2;
    // Two separate diagonal boundaries, one per half of the input space, so a single
    // linear logistic model cannot fit either half without misreading the other.
    const boundary = x1 < 0 ? x1 + x2 : -(x1 - 1) + x2 - 1;
    const p = sigmoid(1.2 * boundary);
    const label = rng.next() < p ? 1 : 0;
    design.push([1, x1, x2]);
    targets.push(label);
  }
  return { design, targets };
}

/**
 * Three genuine regimes along `x`, not two: a distinct linear function on each of
 * `x < -1`, `-1 <= x < 1`, `x >= 1`, same noise level throughout. Two regimes make the
 * gate's soft partition indistinguishable at a glance from any other two-class boundary;
 * three is the smallest case that actually shows what a conditional mixture does,
 * because the reader can see the gate handle more than one decision at once.
 */
export function expertsRegimeChangeData(n = 60): { readonly design: number[][]; readonly t: number[] } {
  const rng = pcg32(EXPERTS_SEED, 1);
  const design: number[][] = [];
  const t: number[] = [];
  const noiseSd = 0.15;
  for (let i = 0; i < n; i++) {
    const x = rng.next() * 6 - 3;
    let mean: number;
    if (x < -1) mean = 1.0 + 0.9 * x;
    else if (x < 1) mean = -1.5 - 0.7 * x;
    else mean = 0.5 + 1.1 * x;
    design.push([1, x]);
    t.push(mean + noiseSd * standardNormal(rng));
  }
  return { design, t };
}
