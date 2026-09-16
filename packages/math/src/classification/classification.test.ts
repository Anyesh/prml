import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import type { Mat, Vec } from '../types.js';
import { argmaxIndex, leastSquaresClassifierWeights, leastSquaresScores } from './leastSquares.js';
import { perceptronTrain } from './perceptron.js';
import { fisherDirection, withinClassScatter } from './fisherLda.js';
import {
  fitSeparateCovarianceGaussian,
  fitSharedCovarianceGaussian,
  posteriorSeparateCovariance,
  posteriorSharedCovariance,
  sharedCovarianceLinearBoundary,
} from './generative.js';
import { crossEntropyError, crossEntropyGradient, irlsFit } from './irls.js';
import { fitLaplaceLogisticPosterior, laplaceApproximation } from './laplace.js';
import { kappa, laplaceLogisticPredictive, logisticGaussianConvolution, probit } from './probit.js';

interface Fixture {
  readonly cases: Record<string, unknown>[];
}

const fixture = loadFixture<Fixture>('classification');
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

function expectMatClose(actual: Mat, expected: readonly (readonly number[])[], tol = TOL) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((row, i) => expectVecClose(actual[i]!, row, tol));
}

function expectClose(actual: number, expected: number, tol = TOL) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));
}

describe('least squares for classification', () => {
  it('matches numpy pinv weights for a three-class 1-of-K fit', () => {
    const c = one('leastSquaresClassifierWeights');
    const got = leastSquaresClassifierWeights(c['design'] as Mat, c['targets'] as Mat);
    expectMatClose(got, c['expected'] as number[][], 1e-9);
  });

  it('matches numpy scores and argmax picks the largest column', () => {
    const c = one('leastSquaresScores');
    const got = leastSquaresScores(c['design'] as Mat, c['weights'] as Mat);
    expectMatClose(got, c['expected'] as number[][], 1e-9);
    expect(argmaxIndex([0.1, 0.9, 0.3])).toBe(1);
    expect(argmaxIndex([5, 1, 2])).toBe(0);
  });
});

describe('perceptron', () => {
  it('reaches the same weights as a fixed-order numpy replication on separable data', () => {
    const c = one('perceptronTrain');
    const got = perceptronTrain(c['design'] as Mat, c['targets'] as (1 | -1)[], {
      initialWeights: c['initialWeights'] as Vec,
      maxEpochs: c['maxEpochs'] as number,
    });
    const expected = c['expected'] as Record<string, unknown>;
    expectVecClose(got.weights, expected['weights'] as number[]);
    expect(got.epochs).toBe(expected['epochs']);
    expect(got.converged).toBe(expected['converged']);
    expect(got.history.map((h) => h.misclassified)).toEqual(expected['history']);
  });

  it('reports non-convergence at the epoch cap on non-separable data', () => {
    const cases = pick('perceptronTrain');
    const c = cases[1]!;
    const got = perceptronTrain(c['design'] as Mat, c['targets'] as (1 | -1)[], {
      initialWeights: c['initialWeights'] as Vec,
      maxEpochs: c['maxEpochs'] as number,
    });
    const expected = c['expected'] as Record<string, unknown>;
    expect(got.converged).toBe(false);
    expect(got.converged).toBe(expected['converged']);
    expectVecClose(got.weights, expected['weights'] as number[]);
  });
});

describe("Fisher's linear discriminant", () => {
  it('matches numpy for the pooled within-class scatter', () => {
    const c = one('withinClassScatter');
    const got = withinClassScatter(c['class1'] as Mat, c['class2'] as Mat, c['mean1'] as Vec, c['mean2'] as Vec);
    expectMatClose(got, c['expected'] as number[][], 1e-9);
  });

  it('matches numpy for the Fisher direction', () => {
    const c = one('fisherDirection');
    const got = fisherDirection(c['mean1'] as Vec, c['mean2'] as Vec, c['within'] as Mat);
    expectVecClose(got, c['expected'] as number[], 1e-9);
  });
});

describe('Gaussian generative classifier', () => {
  it('matches numpy for the shared-covariance maximum-likelihood fit', () => {
    const c = one('fitSharedCovarianceGaussian');
    const got = fitSharedCovarianceGaussian(c['classPoints'] as Mat[]);
    const expected = c['expected'] as Record<string, unknown>;
    expectVecClose(got.priors, expected['priors'] as number[]);
    got.means.forEach((m, k) => expectVecClose(m, (expected['means'] as number[][])[k]!));
    expectMatClose(got.covariance, expected['covariance'] as number[][], 1e-9);
  });

  it('matches numpy for the separate-covariance maximum-likelihood fit', () => {
    const c = one('fitSeparateCovarianceGaussian');
    const got = fitSeparateCovarianceGaussian(c['classPoints'] as Mat[]);
    const expected = c['expected'] as Record<string, unknown>;
    got.covariances.forEach((cov, k) =>
      expectMatClose(cov, (expected['covariances'] as number[][][])[k]!, 1e-9),
    );
  });

  it('matches an independent scipy softmax-of-Gaussian-activations posterior, shared covariance', () => {
    const c = one('posteriorSharedCovariance');
    const fit = fitSharedCovarianceGaussian(c['classPoints'] as Mat[]);
    const expected = c['expected'] as number[][];
    (c['points'] as Vec[]).forEach((x, i) => {
      expectVecClose(posteriorSharedCovariance(x, fit), expected[i]!, 1e-9);
    });
  });

  it('matches an independent scipy softmax-of-Gaussian-activations posterior, separate covariances', () => {
    const c = one('posteriorSeparateCovariance');
    const fit = fitSeparateCovarianceGaussian(c['classPoints'] as Mat[]);
    const expected = c['expected'] as number[][];
    (c['points'] as Vec[]).forEach((x, i) => {
      expectVecClose(posteriorSeparateCovariance(x, fit), expected[i]!, 1e-9);
    });
  });

  it('matches numpy for the closed-form two-class linear boundary', () => {
    const c = one('sharedCovarianceLinearBoundary');
    const fit = fitSharedCovarianceGaussian(c['classPoints'] as Mat[]);
    const got = sharedCovarianceLinearBoundary(fit);
    const expected = c['expected'] as Record<string, unknown>;
    expectVecClose(got.w, expected['w'] as number[], 1e-9);
    expectClose(got.w0, expected['w0'] as number, 1e-9);
  });
});

describe('logistic regression', () => {
  it('matches a softplus-based numpy cross-entropy error at several weight vectors', () => {
    for (const c of pick('crossEntropyError')) {
      const got = crossEntropyError(c['design'] as Mat, c['targets'] as Vec, c['weights'] as Vec);
      expectClose(got, c['expected'] as number, 1e-9);
    }
  });

  it('matches numpy for the cross-entropy gradient', () => {
    for (const c of pick('crossEntropyGradient')) {
      const got = crossEntropyGradient(c['design'] as Mat, c['targets'] as Vec, c['weights'] as Vec);
      expectVecClose(got, c['expected'] as number[], 1e-9);
    }
  });

  it('reaches the same fixed point as a numpy IRLS replication', () => {
    const c = one('irlsFit');
    const got = irlsFit(c['design'] as Mat, c['targets'] as Vec);
    const expected = c['expected'] as Record<string, unknown>;
    expectVecClose(got.weights, expected['weights'] as number[], 1e-9);
    expectMatClose(got.precision, expected['precision'] as number[][], 1e-9);
    expect(got.iterations).toBe(expected['iterations']);
    expect(got.converged).toBe(expected['converged']);
  });
});

describe('Laplace approximation, general form', () => {
  it('matches a numpy Newton iteration on a skewed 1D Gamma-shaped log-density', () => {
    const c = one('laplaceApproximation1D');
    const a = c['shapeParam'] as number;
    const got = laplaceApproximation(
      (x) => [(a - 1) / x[0]! - 1],
      (x) => [[-(a - 1) / (x[0]! * x[0]!)]],
      c['x0'] as Vec,
    );
    const expected = c['expected'] as Record<string, unknown>;
    expectVecClose(got.mode, expected['mode'] as number[], 1e-9);
    expectMatClose(got.covariance, expected['covariance'] as number[][], 1e-9);
    expect(got.iterations).toBe(expected['iterations']);
    expect(got.converged).toBe(expected['converged']);
  });

  it('matches a numpy Newton iteration on an independent 2D Gamma-shaped log-density', () => {
    const c = one('laplaceApproximation2D');
    const [a1, a2] = c['shapeParams'] as [number, number];
    const got = laplaceApproximation(
      (x) => [(a1 - 1) / x[0]! - 1, (a2 - 1) / x[1]! - 1],
      (x) => [
        [-(a1 - 1) / (x[0]! * x[0]!), 0],
        [0, -(a2 - 1) / (x[1]! * x[1]!)],
      ],
      c['x0'] as Vec,
    );
    const expected = c['expected'] as Record<string, unknown>;
    expectVecClose(got.mode, expected['mode'] as number[], 1e-9);
    expectMatClose(got.covariance, expected['covariance'] as number[][], 1e-9);
  });
});

describe('Bayesian logistic regression', () => {
  it('matches a numpy Newton-Raphson MAP fit with a Gaussian prior', () => {
    const c = one('fitLaplaceLogisticPosterior');
    const got = fitLaplaceLogisticPosterior(c['design'] as Mat, c['targets'] as Vec, {
      mean: c['priorMean'] as Vec,
      covariance: c['priorCovariance'] as Mat,
    });
    const expected = c['expected'] as Record<string, unknown>;
    expectVecClose(got.mean, expected['mean'] as number[], 1e-9);
    expectMatClose(got.covariance, expected['covariance'] as number[][], 1e-9);
    expect(got.iterations).toBe(expected['iterations']);
    expect(got.converged).toBe(expected['converged']);
  });
});

describe('probit and the logistic-Gaussian convolution', () => {
  it('matches scipy.stats.norm.cdf', () => {
    const c = one('probit');
    const args = c['args'] as number[];
    const expected = c['expected'] as number[];
    args.forEach((a, i) => expectClose(probit(a), expected[i]!, 1e-10));
  });

  it('matches the closed-form kappa rescaling', () => {
    const c = one('kappa');
    const variances = c['variances'] as number[];
    const expected = c['expected'] as number[];
    variances.forEach((v, i) => expectClose(kappa(v), expected[i]!, 1e-12));
  });

  it('matches an independent scipy convolution approximation', () => {
    const c = one('logisticGaussianConvolution');
    const points = c['points'] as { mean: number; variance: number }[];
    const expected = c['expected'] as number[];
    points.forEach((p, i) => expectClose(logisticGaussianConvolution(p.mean, p.variance), expected[i]!, 1e-10));
  });

  it('matches numpy for the Laplace-posterior predictive probability', () => {
    const c = one('laplaceLogisticPredictive');
    const posterior = { mean: c['posteriorMean'] as Vec, covariance: c['posteriorCovariance'] as Mat };
    const phis = c['phis'] as Vec[];
    const expected = c['expected'] as number[];
    phis.forEach((phi, i) => expectClose(laplaceLogisticPredictive(phi, posterior), expected[i]!, 1e-9));
  });
});
