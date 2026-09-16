import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { hmmGaussianEmissionMatrix, type HmmGaussianComponent } from './hmm.js';
import type { Mat } from '../types.js';

interface Case {
  readonly fn: string;
  readonly data: Mat;
  readonly components: readonly HmmGaussianComponent[];
  readonly expected: Mat;
}

const fixture = loadFixture<{ cases: Case[] }>('hmm_forward_backward');

describe('hmmGaussianEmissionMatrix', () => {
  it('matches scipy multivariate_normal.pdf for every (n, k)', () => {
    const c = fixture.cases.find((c) => c.fn === 'hmmGaussianEmissionMatrix')!;
    const out = hmmGaussianEmissionMatrix(c.data, c.components);
    out.forEach((row, n) => row.forEach((v, k) => expect(v).toBeCloseTo(c.expected[n]![k]!, 9)));
  });
});
