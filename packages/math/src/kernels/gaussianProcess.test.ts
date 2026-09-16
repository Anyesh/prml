import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import { compositeKernel, compositeKernelPartials, type CompositeKernelParams } from './functions.js';
import { fitGPRegression, gpLogMarginalLikelihood, gpLogMarginalLikelihoodGradient, gpPredict, gpPriorSample } from './gaussianProcess.js';

interface GpPredictExpected {
  readonly x: number;
  readonly mean: number;
  readonly variance: number;
}

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

// The fixture builds CN with the same jitter gpFit adds, because leaving it out compares
// two different matrices: at this noise variance the smallest eigenvalue is 0.04, so a
// jitter of 1e-8 moves the answer by roughly 2e-7 and no tolerance short of that would pass.
const TOL = 1e-9;
const close = (actual: number, expected: number, tol = TOL) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));

describe('fitGPRegression and gpPredict', () => {
  it('matches numpy for the predictive mean and variance (6.66-6.67)', () => {
    const c = pick('gpPredict')[0]!;
    const kernel = compositeKernel(c['params'] as CompositeKernelParams);
    const model = fitGPRegression(kernel, trainVecs, T, c['noiseVariance'] as number);
    for (const { x, mean, variance } of c['expected'] as GpPredictExpected[]) {
      const got = gpPredict(model, [x]);
      close(got.mean, mean, 1e-9);
      close(got.variance, variance, 1e-9);
    }
  });

  it('reports growing variance away from the training inputs', () => {
    const c = pick('gpPredict')[0]!;
    const kernel = compositeKernel(c['params'] as CompositeKernelParams);
    const model = fitGPRegression(kernel, trainVecs, T, c['noiseVariance'] as number);
    const near = gpPredict(model, [X[0]!]).variance;
    const far = gpPredict(model, [X[0]! - 5]).variance;
    expect(far).toBeGreaterThan(near);
  });
});

describe('gpLogMarginalLikelihood', () => {
  it('matches numpy (6.69)', () => {
    const c = pick('gpLogMarginalLikelihood')[0]!;
    const kernel = compositeKernel(c['params'] as CompositeKernelParams);
    const model = fitGPRegression(kernel, trainVecs, T, c['noiseVariance'] as number);
    close(gpLogMarginalLikelihood(model), c['expected'] as number, 1e-9);
  });
});

describe('gpLogMarginalLikelihoodGradient', () => {
  it('matches an independent numpy trace-formula derivation (6.70)', () => {
    const c = pick('gpLogMarginalLikelihoodGradient')[0]!;
    const params = c['params'] as CompositeKernelParams;
    const kernel = compositeKernel(params);
    const model = fitGPRegression(kernel, trainVecs, T, c['noiseVariance'] as number);
    const derivatives = compositeKernelPartials(params);
    const gradient = gpLogMarginalLikelihoodGradient(model, derivatives);
    const expected = c['expected'] as number[];
    gradient.forEach((g, i) => close(g, expected[i]!, 1e-9));
  });
});

describe('gpPriorSample', () => {
  it('draws a vector of the right length with no fixture (samplers are checked statistically, see tools/golden/README.md)', () => {
    const rng = pcg32(42);
    const kernel = compositeKernel({ theta0: 1, theta1: 4, theta2: 0, theta3: 0 });
    const xs = trainVecs;
    const sample = gpPriorSample(rng, kernel, xs);
    expect(sample).toHaveLength(xs.length);
  });

  it('has empirical variance near k(x, x) for a repeated point across many draws', () => {
    const rng = pcg32(1234);
    const kernel = compositeKernel({ theta0: 2, theta1: 8, theta2: 0, theta3: 0 });
    const xs = [[0.3]];
    const draws = Array.from({ length: 20000 }, () => gpPriorSample(rng, kernel, xs)[0]!);
    const mean = draws.reduce((s, v) => s + v, 0) / draws.length;
    const variance = draws.reduce((s, v) => s + (v - mean) ** 2, 0) / draws.length;
    expect(Math.abs(variance - kernel(xs[0]!, xs[0]!))).toBeLessThan(0.1);
  });
});
