import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { designMatrix, polynomialBasis, regularisedWeights } from '../regression/index.js';
import { dot } from '../linalg/index.js';
import { rbfKernel } from './functions.js';
import { gramMatrix } from './combine.js';
import { dualPredict, dualRidgeCoefficients, nadarayaWatsonPredict, nadarayaWatsonWeights } from './dual.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

interface Fixture {
  readonly dataset1d: { readonly x: number[]; readonly t: number[]; readonly testX: number[] };
  readonly cases: Case[];
}

const fixture = loadFixture<Fixture>('kernels');
const pick = (fn: string) => fixture.cases.filter((c) => c.fn === fn);
const { x: X, t: T } = fixture.dataset1d;
const trainVecs = X.map((xi) => [xi]);

const TOL = 1e-9;
const close = (actual: number, expected: number, tol = TOL) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));

describe('dualRidgeCoefficients and dualPredict', () => {
  it('matches numpy for a = (K + lambda I)^-1 t over an RBF kernel', () => {
    for (const c of pick('dualRidgeCoefficients')) {
      const kernel = rbfKernel(c['lengthScale'] as number);
      const gram = gramMatrix(kernel, trainVecs);
      const a = dualRidgeCoefficients(gram, T, c['lambda'] as number);
      const expected = c['expected'] as number[];
      a.forEach((v, i) => close(v, expected[i]!));
    }
  });

  it('matches numpy for predictions at unseen test points', () => {
    for (const c of pick('dualPredict')) {
      const kernel = rbfKernel(c['lengthScale'] as number);
      const gram = gramMatrix(kernel, trainVecs);
      const a = dualRidgeCoefficients(gram, T, c['lambda'] as number);
      const testX = c['testX'] as number[];
      const expected = c['expected'] as number[];
      testX.forEach((xt, i) => close(dualPredict(kernel, trainVecs, a, [xt]), expected[i]!));
    }
  });

  it('agrees with weight-space ridge regression through an explicit polynomial feature map (6.8-6.9)', () => {
    const degree = 3;
    const lambda = 0.02;
    const phi = polynomialBasis(degree, { bias: true });
    const design = designMatrix(X, phi);
    const primalWeights = regularisedWeights(design, T, lambda);

    const featureMapKernel = (a: readonly number[], b: readonly number[]) => dot(phi(a[0]!), phi(b[0]!));
    const gram = gramMatrix(featureMapKernel, trainVecs);
    const dualCoefficients = dualRidgeCoefficients(gram, T, lambda);

    for (const xt of fixture.dataset1d.testX) {
      const primalPrediction = dot(phi(xt), primalWeights);
      const dualPrediction = dualPredict(featureMapKernel, trainVecs, dualCoefficients, [xt]);
      close(dualPrediction, primalPrediction);
    }
  });
});

describe('nadarayaWatsonWeights and nadarayaWatsonPredict', () => {
  it('weights sum to one for every query point (the constraint under 6.47)', () => {
    for (const c of pick('nadarayaWatson')) {
      const kernel = rbfKernel(c['bandwidth'] as number);
      for (const xt of c['testX'] as number[]) {
        const weights = nadarayaWatsonWeights(kernel, trainVecs, [xt]);
        close(weights.reduce((s, w) => s + w, 0), 1);
      }
    }
  });

  it('matches numpy for the weights and the predicted mean (6.45-6.46)', () => {
    for (const c of pick('nadarayaWatson')) {
      const kernel = rbfKernel(c['bandwidth'] as number);
      const testX = c['testX'] as number[];
      const expectedWeights = c['expectedWeights'] as number[][];
      const expectedPredictions = c['expectedPredictions'] as number[];
      testX.forEach((xt, i) => {
        const weights = nadarayaWatsonWeights(kernel, trainVecs, [xt]);
        weights.forEach((w, n) => close(w, expectedWeights[i]![n]!));
        close(nadarayaWatsonPredict(kernel, trainVecs, T, [xt]), expectedPredictions[i]!);
      });
    }
  });
});
