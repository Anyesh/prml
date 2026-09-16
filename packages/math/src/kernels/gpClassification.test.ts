import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { rbfKernel } from './functions.js';
import { fitGPClassificationLaplace, gpClassificationPredict } from './gpClassification.js';

interface ClassificationExpected {
  readonly x: number[];
  readonly meanA: number;
  readonly varianceA: number;
  readonly probability: number;
}

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

interface Fixture {
  readonly classificationDataset: { readonly points: number[][]; readonly targets: number[] };
  readonly cases: Case[];
}

const fixture = loadFixture<Fixture>('kernels');
const pick = (fn: string) => fixture.cases.filter((c) => c.fn === fn);
const { points, targets } = fixture.classificationDataset;

const TOL = 1e-9;
const close = (actual: number, expected: number, tol = TOL) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));

describe('fitGPClassificationLaplace', () => {
  it('matches an independent numpy Newton solve for the posterior mode a* (6.77-6.84)', () => {
    const c = pick('gpClassificationMode')[0]!;
    const kernel = rbfKernel(c['lengthScale'] as number);
    const model = fitGPClassificationLaplace(kernel, points, targets, c['nu'] as number);
    const expected = c['expected'] as number[];
    model.mode.forEach((v, i) => close(v, expected[i]!));
    expect(model.converged).toBe(true);
  });

  it('places every mode value on the same side of zero as its label', () => {
    const c = pick('gpClassificationMode')[0]!;
    const kernel = rbfKernel(c['lengthScale'] as number);
    const model = fitGPClassificationLaplace(kernel, points, targets, c['nu'] as number);
    model.mode.forEach((a, i) => expect(Math.sign(a)).toBe(targets[i] === 1 ? 1 : -1));
  });
});

describe('gpClassificationPredict', () => {
  it('matches an independent numpy derivation of the predictive mean, variance and squashed probability (6.87-6.88, 4.153)', () => {
    const c = pick('gpClassificationPredict')[0]!;
    const kernel = rbfKernel(c['lengthScale'] as number);
    const model = fitGPClassificationLaplace(kernel, points, targets, c['nu'] as number);
    const testPoints = c['testPoints'] as number[][];
    const expected = c['expected'] as ClassificationExpected[];
    testPoints.forEach((x, i) => {
      const got = gpClassificationPredict(model, x);
      close(got.meanA, expected[i]!.meanA);
      close(got.varianceA, expected[i]!.varianceA);
      close(got.probability, expected[i]!.probability);
    });
  });

  it('keeps every probability inside (0, 1)', () => {
    const c = pick('gpClassificationPredict')[0]!;
    const kernel = rbfKernel(c['lengthScale'] as number);
    const model = fitGPClassificationLaplace(kernel, points, targets, c['nu'] as number);
    for (const x of c['testPoints'] as number[][]) {
      const { probability } = gpClassificationPredict(model, x);
      expect(probability).toBeGreaterThan(0);
      expect(probability).toBeLessThan(1);
    }
  });
});
