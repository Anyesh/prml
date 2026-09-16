import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import type { Mat, Vec } from '../types.js';
import { centerGramMatrix, kernelPcaGramMatrix, kernelPcaFit, kernelPcaProject, kernelPcaTrainingProjections, kernelPcaRbfKernel } from './kernelPca.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('kernelPca');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('kernelPcaGramMatrix / kernelPcaRbfKernel', () => {
  it('matches an independently computed RBF Gram matrix', () => {
    const c = casesFor('gramMatrix')[0]!;
    const data = c['data'] as Mat;
    const kernel = kernelPcaRbfKernel(c['gamma'] as number);
    const result = kernelPcaGramMatrix(data, kernel);
    const expected = c['expected'] as number[][];
    result.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('centerGramMatrix', () => {
  it('matches PRML 12.83-12.85, as if the feature vectors had zero mean', () => {
    const c = casesFor('centerGramMatrix')[0]!;
    const result = centerGramMatrix(c['k'] as Mat);
    const expected = c['expected'] as number[][];
    result.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('kernelPcaFit', () => {
  it('solves the reduced eigenproblem (PRML 12.80) normalised per 12.81', () => {
    const c = casesFor('kernelPcaFit')[0]!;
    const data = c['data'] as Mat;
    const kernel = kernelPcaRbfKernel(c['gamma'] as number);
    const model = kernelPcaFit(data, kernel, c['numComponents'] as number);
    const expected = c['expected'] as { eigenvalues: number[]; alphas: number[][] };
    model.eigenvalues.forEach((v, i) => expect(v).toBeCloseTo(expected.eigenvalues[i]!, 9));
    model.alphas.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected.alphas[i]![j]!, 9)));
  });

  it('matches training-set projections computed directly from the centred Gram matrix', () => {
    const c = casesFor('kernelPcaTrainingProjections')[0]!;
    const data = c['data'] as Mat;
    const kernel = kernelPcaRbfKernel(c['gamma'] as number);
    const model = kernelPcaFit(data, kernel, c['numComponents'] as number);
    const result = kernelPcaTrainingProjections(model);
    const expected = c['expected'] as number[][];
    result.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('projects a held-out point via the centred kernel vector (PRML 12.82)', () => {
    const c = casesFor('kernelPcaProject')[0]!;
    const data = c['data'] as Mat;
    const kernel = kernelPcaRbfKernel(c['gamma'] as number);
    const model = kernelPcaFit(data, kernel, c['numComponents'] as number);
    const result = kernelPcaProject(model, c['query'] as Vec);
    const expected = c['expected'] as number[];
    result.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
  });
});
