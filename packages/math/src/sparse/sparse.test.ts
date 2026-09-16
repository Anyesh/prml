import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import type { Mat, Vec } from '../types.js';
import {
  epsilonInsensitiveLoss,
  hingeLoss,
  logisticMarginLoss,
  misclassificationLoss,
  smoFitClassifier,
  smoFitRegression,
  solveBoxConstrainedQp,
  sparseLinearKernel,
  sparsePolynomialKernel,
  sparseRbfKernel,
  squaredMarginLoss,
  svmDecisionFunction,
  svrPredictFunction,
} from './index.js';
import {
  rvmClassificationFit,
  rvmOptimalSingleAlpha,
  rvmRegressionFit,
  rvmRegressionPredictive,
  rvmSingleAlphaLogEvidenceTerm,
  rvmSparsityStatistics,
} from './rvm.js';

interface Fixture {
  readonly cases: Record<string, unknown>[];
}

const fixture = loadFixture<Fixture>('sparse');
const pick = (fn: string) => fixture.cases.filter((c) => c['fn'] === fn);
const one = (fn: string) => {
  const found = pick(fn)[0];
  if (!found) throw new Error(`fixture has no case for ${fn}`);
  return found;
};

const TOL = 1e-9;

function expectClose(actual: number, expected: number, tol = TOL) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));
}

function expectVecClose(actual: readonly number[], expected: readonly number[], tol = TOL) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((e, i) => expectClose(actual[i]!, e, tol));
}

function expectMatClose(actual: Mat, expected: readonly (readonly number[])[], tol = TOL) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((row, i) => expectVecClose(actual[i]!, row, tol));
}

describe('kernels', () => {
  it('matches numpy for the linear kernel', () => {
    for (const c of pick('sparseLinearKernel')) {
      expectClose(sparseLinearKernel(c['a'] as Vec, c['b'] as Vec), c['expected'] as number);
    }
  });

  it('matches numpy for the polynomial kernel', () => {
    for (const c of pick('sparsePolynomialKernel')) {
      const kernel = sparsePolynomialKernel({ degree: c['degree'] as number, offset: c['offset'] as number });
      expectClose(kernel(c['a'] as Vec, c['b'] as Vec), c['expected'] as number);
    }
  });

  it('matches numpy for the RBF kernel', () => {
    for (const c of pick('sparseRbfKernel')) {
      const kernel = sparseRbfKernel(c['gamma'] as number);
      expectClose(kernel(c['a'] as Vec, c['b'] as Vec), c['expected'] as number);
    }
  });
});

describe('margin-based losses (Figure 7.5)', () => {
  it('hinge loss matches [1-z]+', () => {
    const c = one('hingeLoss');
    const margins = c['margins'] as number[];
    const expected = c['expected'] as number[];
    margins.forEach((z, i) => expectClose(hingeLoss(z), expected[i]!));
  });

  it('rescaled logistic loss matches ln(1+exp(-z))/ln2', () => {
    const c = one('logisticMarginLoss');
    const margins = c['margins'] as number[];
    const expected = c['expected'] as number[];
    margins.forEach((z, i) => expectClose(logisticMarginLoss(z), expected[i]!));
  });

  it('squared margin loss matches (1-z)^2', () => {
    const c = one('squaredMarginLoss');
    const margins = c['margins'] as number[];
    const expected = c['expected'] as number[];
    margins.forEach((z, i) => expectClose(squaredMarginLoss(z), expected[i]!));
  });

  it('misclassification loss is the 0/1 step at z=0', () => {
    const c = one('misclassificationLoss');
    const margins = c['margins'] as number[];
    const expected = c['expected'] as number[];
    margins.forEach((z, i) => expect(misclassificationLoss(z)).toBe(expected[i]));
  });

  it('epsilon-insensitive loss is zero inside the tube', () => {
    const c = one('epsilonInsensitiveLoss');
    const residuals = c['residuals'] as number[];
    const epsilons = c['epsilons'] as number[];
    const expected = c['expected'] as number[];
    residuals.forEach((r, i) => expectClose(epsilonInsensitiveLoss(r, epsilons[i]!), expected[i]!));
  });
});

describe('SMO for classification (7.10-7.18, 7.32-7.37)', () => {
  for (const c of pick('smoFitClassifier')) {
    it(`matches the exact KKT solution on the "${c['name']}" case`, () => {
      const points = c['points'] as Mat;
      const labels = (c['labels'] as number[]).map((y) => y as 1 | -1);
      const C = c['C'] as number;
      const expected = c['expected'] as { alpha: number[]; bias: number; decision: number[] };

      const fit = smoFitClassifier(points, labels, sparseLinearKernel, { C, tol: 1e-13, maxIterations: 20_000 });
      expectVecClose(fit.alpha, expected.alpha, 1e-9);
      expectClose(fit.bias, expected.bias, 1e-9);

      const decision = svmDecisionFunction(fit, points, labels, sparseLinearKernel);
      const probe = c['probe'] as Mat;
      const decisionValues = probe.map((x) => decision(x));
      expectVecClose(decisionValues, expected.decision, 1e-9);

      // Every alpha lies in the box and the equality constraint holds at solver precision,
      // independent of what the fixture asserts.
      for (const a of fit.alpha) {
        expect(a).toBeGreaterThanOrEqual(-1e-9);
        expect(a).toBeLessThanOrEqual(C + 1e-9);
      }
      const equality = fit.alpha.reduce((s, a, i) => s + a * labels[i]!, 0);
      expect(Math.abs(equality)).toBeLessThan(1e-9);
    });
  }
});

describe('SMO for regression (7.50-7.69)', () => {
  for (const c of pick('smoFitRegression')) {
    it(`matches the exact KKT solution on the "${c['name']}" case`, () => {
      const points = c['points'] as Mat;
      const targets = c['targets'] as Vec;
      const C = c['C'] as number;
      const epsilon = c['epsilon'] as number;
      const expected = c['expected'] as { coefficients: number[]; bias: number; predictions: number[] };

      const fit = smoFitRegression(points, targets, sparseLinearKernel, { C, epsilon, tol: 1e-13, maxIterations: 20_000 });
      expectVecClose(fit.coefficients, expected.coefficients, 1e-9);
      expectClose(fit.bias, expected.bias, 1e-9);

      const predict = svrPredictFunction(fit, points, sparseLinearKernel);
      const probe = c['probe'] as Mat;
      expectVecClose(
        probe.map((x) => predict(x)),
        expected.predictions,
        1e-9,
      );

      for (const coeff of fit.coefficients) {
        expect(Math.abs(coeff)).toBeLessThanOrEqual(C + 1e-9);
      }
    });
  }
});

describe('solveBoxConstrainedQp', () => {
  it('solves a two-variable problem to its closed-form optimum', () => {
    // Q = 2I, p = [1,1], y = [1,-1]: the equality constraint forces a1 = a2 = a, reducing
    // the objective to 2a^2 - 2a over a in [0,5], minimised at a = 0.5.
    const result = solveBoxConstrainedQp({
      q: [
        [2, 0],
        [0, 2],
      ],
      p: [1, 1],
      y: [1, -1],
      upperBound: [5, 5],
    });
    expect(result.converged).toBe(true);
    expectClose(result.alpha[0]!, 0.5);
    expectClose(result.alpha[1]!, 0.5);
  });
});

describe('RVM regression (7.82-7.91)', () => {
  it('matches the numpy replication of the direct re-estimation fixed point', () => {
    const c = one('rvmRegressionFit');
    const design = c['design'] as Mat;
    const targets = c['targets'] as Vec;
    const initialAlpha = c['initialAlpha'] as Vec;
    const initialBeta = c['initialBeta'] as number;
    const expected = c['expected'] as {
      alpha: number[];
      beta: number;
      mean: number[];
      covariance: number[][];
      gamma: number[];
      iterations: number;
      converged: boolean;
    };

    const fit = rvmRegressionFit(design, targets, initialAlpha, initialBeta);
    expectVecClose(fit.alpha, expected.alpha, 1e-9);
    expectClose(fit.beta, expected.beta, 1e-9);
    expectVecClose(fit.mean, expected.mean, 1e-9);
    expectMatClose(fit.covariance, expected.covariance, 1e-9);
    expectVecClose(fit.gamma, expected.gamma, 1e-9);
    // Not an exact iteration count: JS and numpy's libm implementations of exp/log
    // differ in their last bit, which occasionally shifts which sweep first satisfies a
    // tol=1e-12 relative convergence test by one, with no effect on the converged values
    // above. The final fixed point is what golden fixtures exist to pin down, not the
    // exact step count that reached it.
    expect(Math.abs(fit.iterations - expected.iterations)).toBeLessThanOrEqual(2);
    expect(fit.converged).toBe(expected.converged);
  });

  it('matches the predictive mean and variance (7.90-7.91)', () => {
    const c = one('rvmRegressionPredictive');
    const phis = c['phis'] as Mat;
    const mean = c['mean'] as Vec;
    const covariance = c['covariance'] as Mat;
    const beta = c['beta'] as number;
    const expected = c['expected'] as { mean: number[]; variance: number[] };

    const fit = { mean: [...mean], covariance: covariance.map((r) => [...r]), precision: covariance, beta } as Parameters<
      typeof rvmRegressionPredictive
    >[1];
    const predictions = phis.map((phi) => rvmRegressionPredictive(phi, fit));
    expectVecClose(
      predictions.map((p) => p.mean),
      expected.mean,
      1e-9,
    );
    expectVecClose(
      predictions.map((p) => p.variance),
      expected.variance,
      1e-9,
    );
  });
});

describe('RVM sparsity analysis (7.92-7.107)', () => {
  it('matches the closed-form Q, S, q, s at a fixed (alpha, beta)', () => {
    const c = one('rvmSparsityStatistics');
    const design = c['design'] as Mat;
    const targets = c['targets'] as Vec;
    const alpha = c['alpha'] as Vec;
    const beta = c['beta'] as number;
    const expected = c['expected'] as { Q: number[]; S: number[]; q: number[]; s: number[] };

    const stats = rvmSparsityStatistics(design, targets, alpha, beta);
    expectVecClose(stats.Q, expected.Q, 1e-9);
    expectVecClose(stats.S, expected.S, 1e-9);
    expectVecClose(stats.q, expected.q, 1e-9);
    expectVecClose(stats.s, expected.s, 1e-9);
  });

  it('matches lambda(alpha) (7.97) across a grid', () => {
    for (const c of pick('rvmSingleAlphaLogEvidenceTerm')) {
      const s = c['s'] as number;
      const q = c['q'] as number;
      const alphaGrid = c['alphaGrid'] as number[];
      const expected = c['expected'] as number[];
      alphaGrid.forEach((a, i) => expectClose(rvmSingleAlphaLogEvidenceTerm(a, s, q), expected[i]!));
    }
  });

  it('matches the stationary point (7.100-7.101), including the alpha=Infinity case', () => {
    for (const c of pick('rvmOptimalSingleAlpha')) {
      const s = c['s'] as number;
      const q = c['q'] as number;
      const expected = c['expected'] as number | null;
      const got = rvmOptimalSingleAlpha(s, q);
      if (expected === null) {
        expect(got).toBe(Infinity);
      } else {
        expectClose(got, expected);
      }
    }
  });
});

describe('RVM classification (7.108-7.119)', () => {
  it('matches the numpy replication reusing the same ARD re-estimation fixed point', () => {
    const c = one('rvmClassificationFit');
    const design = c['design'] as Mat;
    const targets = c['targets'] as Vec;
    const initialAlpha = c['initialAlpha'] as Vec;
    const expected = c['expected'] as {
      alpha: number[];
      mean: number[];
      covariance: number[][];
      gamma: number[];
      iterations: number;
      converged: boolean;
    };

    const fit = rvmClassificationFit(design, targets, initialAlpha);
    expectVecClose(fit.alpha, expected.alpha, 1e-9);
    expectVecClose(fit.mean, expected.mean, 1e-9);
    expectMatClose(fit.covariance, expected.covariance, 1e-9);
    expectVecClose(fit.gamma, expected.gamma, 1e-9);
    // See the matching note in the regression fit test above: cross-language libm
    // rounding can shift the exact sweep a tol=1e-10 stopping test fires on by one.
    expect(Math.abs(fit.iterations - expected.iterations)).toBeLessThanOrEqual(2);
    expect(fit.converged).toBe(expected.converged);
  });
});
