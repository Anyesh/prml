import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  compositeKernel,
  compositeKernelPartials,
  exponentialKernel,
  linearKernel,
  polynomialKernel,
  rbfKernel,
  sigmoidKernel,
  type CompositeKernelParams,
} from './functions.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

interface Fixture {
  readonly cases: Case[];
}

const fixture = loadFixture<Fixture>('kernels');
const pick = (fn: string) => fixture.cases.filter((c) => c.fn === fn);

const TOL = 1e-9;
const close = (actual: number, expected: number, tol = TOL) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));

describe('linearKernel', () => {
  it('matches numpy for the dot-product kernel in 2-D and 3-D', () => {
    const k = linearKernel();
    for (const c of pick('linearKernel')) {
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });
});

describe('polynomialKernel', () => {
  it('matches numpy, including the (x^T z)^2 expansion PRML 6.12 walks through by hand', () => {
    for (const c of pick('polynomialKernel')) {
      const k = polynomialKernel(c['degree'] as number, c['c'] as number);
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });
});

describe('rbfKernel', () => {
  it('matches numpy for the isotropic Gaussian kernel (6.23)', () => {
    for (const c of pick('rbfKernel').filter((c) => typeof c['lengthScale'] === 'number')) {
      const k = rbfKernel(c['lengthScale'] as number);
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });

  it('matches numpy for the ARD form (6.71) with one length scale per axis', () => {
    for (const c of pick('rbfKernel').filter((c) => Array.isArray(c['lengthScale']))) {
      const k = rbfKernel(c['lengthScale'] as number[]);
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });
});

describe('exponentialKernel', () => {
  it('matches numpy for the Ornstein-Uhlenbeck kernel (6.56)', () => {
    for (const c of pick('exponentialKernel')) {
      const k = exponentialKernel(c['theta'] as number);
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });
});

describe('compositeKernel', () => {
  it('matches numpy at the exact (theta0..theta3) tuples PRML plots in Figure 6.5', () => {
    for (const c of pick('compositeKernel')) {
      const k = compositeKernel(c['params'] as CompositeKernelParams);
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });
});

describe('compositeKernelPartials', () => {
  it('matches an independent numpy derivation, itself checked there against finite differences', () => {
    for (const c of pick('compositeKernelPartials')) {
      const partials = compositeKernelPartials(c['params'] as CompositeKernelParams);
      const expected = c['expected'] as number[];
      partials.forEach((p, i) => close(p(c['x'] as number[], c['xp'] as number[]), expected[i]!));
    }
  });
});

describe('sigmoidKernel', () => {
  it('matches numpy for the tanh kernel (6.37)', () => {
    for (const c of pick('sigmoidKernel')) {
      const k = sigmoidKernel(c['a'] as number, c['b'] as number);
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });
});
