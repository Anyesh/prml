import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { rbfKernel, linearKernel } from './functions.js';
import { gramMatrix, kernelVector, productKernel, scaleKernel, sumKernel } from './combine.js';

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
const close = (actual: number, expected: number) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(TOL * Math.max(1, Math.abs(expected)));

describe('gramMatrix', () => {
  it('matches numpy for an RBF kernel over four points', () => {
    const c = pick('gramMatrix')[0]!;
    const points = c['points'] as number[][];
    const k = rbfKernel(c['lengthScale'] as number);
    const gram = gramMatrix(k, points);
    const expected = c['expected'] as number[][];
    gram.forEach((row, i) => row.forEach((v, j) => close(v, expected[i]![j]!)));
  });

  it('is symmetric by construction', () => {
    const c = pick('gramMatrix')[0]!;
    const points = c['points'] as number[][];
    const k = rbfKernel(c['lengthScale'] as number);
    const gram = gramMatrix(k, points);
    gram.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(gram[j]![i]!, 12)));
  });
});

describe('kernelVector', () => {
  it('is the query column of the corresponding Gram matrix', () => {
    const c = pick('gramMatrix')[0]!;
    const points = c['points'] as number[][];
    const k = rbfKernel(c['lengthScale'] as number);
    const vec = kernelVector(k, points, points[2]!);
    const gram = gramMatrix(k, points);
    vec.forEach((v, i) => close(v, gram[i]![2]!));
  });
});

describe('sumKernel, productKernel, scaleKernel', () => {
  it('matches numpy for the sum of a linear and an RBF kernel (6.17)', () => {
    const k = sumKernel(linearKernel(), rbfKernel(1.0));
    for (const c of pick('sumKernel')) {
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });

  it('matches numpy for the product of a linear and an RBF kernel (6.18)', () => {
    const k = productKernel(linearKernel(), rbfKernel(1.0));
    for (const c of pick('productKernel')) {
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });

  it('matches numpy for scaling a linear kernel by a positive constant (6.13)', () => {
    for (const c of pick('scaleKernel')) {
      const k = scaleKernel(linearKernel(), c['c'] as number);
      close(k(c['x'] as number[], c['xp'] as number[]), c['expected'] as number);
    }
  });
});
