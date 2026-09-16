import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import {
  designMatrix,
  gaussianBasis,
  maximumLikelihoodWeights,
  meanSquaredError,
  polynomialBasis,
  regularisedWeights,
  sigmoidalBasis,
} from './basis.js';
import {
  equivalentKernel,
  isotropicPrior,
  predictive,
  sampleWeights,
  updatePosterior,
  weightPosterior,
} from './bayesianLinear.js';
import { logEvidence, maximiseEvidence } from './evidence.js';
import { biasVarianceDecomposition, ensembleMean } from './biasVariance.js';

interface Fixture {
  readonly dataset: { readonly x: number[]; readonly t: number[]; readonly noiseSd: number };
  readonly centres: number[];
  readonly scale: number;
  readonly cases: Record<string, unknown>[];
}

const fixture = loadFixture<Fixture>('regression');
const { x: X, t: T } = fixture.dataset;

const pick = (fn: string) => fixture.cases.filter((c) => c['fn'] === fn);
const one = (fn: string) => {
  const found = pick(fn)[0];
  if (!found) throw new Error(`fixture has no case for ${fn}`);
  return found;
};

const TOL = 1e-9;

function expectVecClose(actual: readonly number[], expected: readonly number[], tol = TOL) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((e, i) => {
    expect(Math.abs(actual[i]! - e)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(e)));
  });
}

function expectMatClose(
  actual: readonly (readonly number[])[],
  expected: readonly (readonly number[])[],
  tol = TOL,
) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((row, i) => expectVecClose(actual[i]!, row, tol));
}

describe('basis functions', () => {
  it('matches numpy for the polynomial basis, with and without the bias feature', () => {
    for (const c of pick('polynomialBasis')) {
      const phi = polynomialBasis(c['degree'] as number, { bias: c['bias'] as boolean });
      expectVecClose(phi(c['x'] as number), c['expected'] as number[]);
    }
  });

  it('matches numpy for the gaussian basis', () => {
    for (const c of pick('gaussianBasis')) {
      const phi = gaussianBasis(c['centres'] as number[], c['scale'] as number, { bias: true });
      expectVecClose(phi(c['x'] as number), c['expected'] as number[]);
    }
  });

  it('matches numpy for the sigmoidal basis', () => {
    for (const c of pick('sigmoidalBasis')) {
      const phi = sigmoidalBasis(c['centres'] as number[], c['scale'] as number, { bias: true });
      expectVecClose(phi(c['x'] as number), c['expected'] as number[]);
    }
  });

  it('builds a design matrix with one row per input', () => {
    const design = designMatrix(X, polynomialBasis(3));
    expect(design).toHaveLength(X.length);
    expect(design[0]).toHaveLength(4);
  });
});

describe('least squares', () => {
  it('matches numpy weights, including the ill-conditioned degree-9 fit', () => {
    for (const c of pick('maximumLikelihoodWeights')) {
      const design = designMatrix(X, polynomialBasis(c['degree'] as number));
      expectVecClose(maximumLikelihoodWeights(design, T), c['expected'] as number[], 1e-6);
    }
  });

  it('matches numpy for the regularised solution across four decades of lambda', () => {
    for (const c of pick('regularisedWeights')) {
      const design = designMatrix(X, polynomialBasis(c['degree'] as number));
      expectVecClose(
        regularisedWeights(design, T, c['lambda'] as number),
        c['expected'] as number[],
        1e-6,
      );
    }
  });

  it('matches numpy for the mean squared error', () => {
    for (const c of pick('meanSquaredError')) {
      const design = designMatrix(X, polynomialBasis(c['degree'] as number));
      const actual = meanSquaredError(design, T, c['weights'] as number[]);
      expect(Math.abs(actual - (c['expected'] as number))).toBeLessThanOrEqual(TOL);
    }
  });
});

describe('Bayesian linear regression', () => {
  it('matches numpy for the weight posterior mean, covariance and precision', () => {
    for (const c of pick('weightPosterior')) {
      const design = designMatrix(X, polynomialBasis(c['degree'] as number));
      const post = weightPosterior(design, T, {
        alpha: c['alpha'] as number,
        beta: c['beta'] as number,
      });
      const expected = c['expected'] as Record<string, number[][] | number[]>;
      expectVecClose(post.mean, expected['mean'] as number[], 1e-7);
      expectMatClose(post.cov, expected['cov'] as number[][], 1e-7);
      expectMatClose(post.precision, expected['precision'] as number[][], 1e-9);
    }
  });

  it('matches numpy for the predictive mean and variance, including outside the data range', () => {
    for (const c of pick('predictive')) {
      const design = designMatrix(X, polynomialBasis(c['degree'] as number));
      const beta = c['beta'] as number;
      const post = weightPosterior(design, T, { alpha: c['alpha'] as number, beta });
      const phi = polynomialBasis(c['degree'] as number);
      const got = predictive(phi(c['x'] as number), post, beta);
      const expected = c['expected'] as Record<string, number>;
      expect(Math.abs(got.mean - expected['mean']!)).toBeLessThanOrEqual(
        1e-7 * Math.max(1, Math.abs(expected['mean']!)),
      );
      expect(Math.abs(got.variance - expected['variance']!)).toBeLessThanOrEqual(
        1e-7 * Math.max(1, Math.abs(expected['variance']!)),
      );
    }
  });

  it('reaches the batch posterior by folding points in one at a time', () => {
    for (const c of pick('updatePosterior')) {
      const degree = c['degree'] as number;
      const beta = c['beta'] as number;
      const count = c['count'] as number;
      const phi = polynomialBasis(degree);

      let post = isotropicPrior(degree + 1, c['alpha'] as number);
      for (let n = 0; n < count; n++) post = updatePosterior(post, phi(X[n]!), T[n]!, beta);

      const expected = c['expected'] as Record<string, number[] | number[][]>;
      expectVecClose(post.mean, expected['mean'] as number[], 1e-7);
      expectMatClose(post.cov, expected['cov'] as number[][], 1e-7);
    }
  });

  it('matches numpy for the equivalent kernel', () => {
    const c = one('equivalentKernel');
    const centres = c['centres'] as number[];
    const scale = c['scale'] as number;
    const beta = c['beta'] as number;
    const phi = gaussianBasis(centres, scale, { bias: true });
    const post = weightPosterior(designMatrix(X, phi), T, { alpha: c['alpha'] as number, beta });
    const k = equivalentKernel(post, beta);

    for (const row of c['rows'] as Record<string, number | number[]>[]) {
      const probe = row['probe'] as number[];
      const got = probe.map((xb) => k(phi(row['x'] as number), phi(xb)));
      expectVecClose(got, row['expected'] as number[], 1e-7);
    }
  });

  it('draws weight samples whose empirical moments recover the posterior', () => {
    const design = designMatrix(X, polynomialBasis(3));
    const post = weightPosterior(design, T, { alpha: 2e-3, beta: 25 });
    const draws = sampleWeights(pcg32(11, 3), post, 60_000);
    expect(draws).toHaveLength(60_000);

    // Six standard errors of the sample mean, which is the tolerance a correct sampler
    // clears essentially always and a wrong one misses by orders of magnitude.
    post.mean.forEach((mu, j) => {
      const empirical = draws.reduce((s, w) => s + w[j]!, 0) / draws.length;
      const stderr = Math.sqrt(post.cov[j]![j]! / draws.length);
      expect(Math.abs(empirical - mu)).toBeLessThanOrEqual(6 * stderr);
    });
  });
});

describe('evidence', () => {
  it('matches numpy for the log marginal likelihood across model orders', () => {
    for (const c of pick('logEvidence')) {
      const design = designMatrix(X, polynomialBasis(c['degree'] as number));
      const got = logEvidence(design, T, { alpha: c['alpha'] as number, beta: c['beta'] as number });
      const expected = c['expected'] as number;
      expect(Math.abs(got - expected)).toBeLessThanOrEqual(1e-7 * Math.max(1, Math.abs(expected)));
    }
  });

  it('converges to numpy hyperparameters and effective parameter count', () => {
    for (const c of pick('maximiseEvidence')) {
      const design = designMatrix(X, polynomialBasis(c['degree'] as number));
      const got = maximiseEvidence(design, T, {
        alpha: c['initialAlpha'] as number,
        beta: c['initialBeta'] as number,
      });
      const expected = c['expected'] as Record<string, number>;
      expect(got.converged).toBe(true);
      for (const key of ['alpha', 'beta', 'effectiveParameters', 'logEvidence'] as const) {
        const want = expected[key]!;
        expect(Math.abs(got[key] - want)).toBeLessThanOrEqual(1e-6 * Math.max(1, Math.abs(want)));
      }
    }
  });

  it('never reports more effective parameters than the model has weights', () => {
    const design = designMatrix(X, polynomialBasis(9));
    const got = maximiseEvidence(design, T, { alpha: 1, beta: 1 });
    expect(got.effectiveParameters).toBeGreaterThan(0);
    expect(got.effectiveParameters).toBeLessThanOrEqual(10);
  });
});

describe('bias-variance', () => {
  it('matches numpy across three regularisation strengths', () => {
    for (const c of pick('biasVarianceDecomposition')) {
      const got = biasVarianceDecomposition(
        c['predictions'] as number[][],
        c['truth'] as number[],
      );
      const expected = c['expected'] as Record<string, number>;
      for (const key of ['bias2', 'variance', 'total'] as const) {
        expect(Math.abs(got[key] - expected[key]!)).toBeLessThanOrEqual(
          TOL * Math.max(1, Math.abs(expected[key]!)),
        );
      }
    }
  });

  it('trades bias against variance as regularisation increases', () => {
    const byLambda = pick('biasVarianceDecomposition')
      .map((c) => ({
        lambda: c['lambda'] as number,
        result: biasVarianceDecomposition(c['predictions'] as number[][], c['truth'] as number[]),
      }))
      .sort((a, b) => a.lambda - b.lambda);

    for (let i = 1; i < byLambda.length; i++) {
      expect(byLambda[i]!.result.variance).toBeLessThan(byLambda[i - 1]!.result.variance);
      expect(byLambda[i]!.result.bias2).toBeGreaterThan(byLambda[i - 1]!.result.bias2);
    }
  });

  it('averages the ensemble pointwise', () => {
    const mean = ensembleMean([
      [1, 2, 3],
      [3, 4, 5],
    ]);
    expectVecClose(mean, [2, 3, 4]);
  });
});
