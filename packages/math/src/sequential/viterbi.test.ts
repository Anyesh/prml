import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { hmmViterbi } from './viterbi.js';
import { hmmGammaScaled, hmmForwardScaled, hmmBackwardScaled } from './forwardBackward.js';
import type { Mat, Vec } from '../types.js';

interface Case {
  readonly fn: string;
  readonly pi: Vec;
  readonly A: Mat;
  readonly B: Mat;
  readonly expected: { omega: Mat; psi: number[][]; path: number[]; logProb: number };
}

const fixture = loadFixture<{ cases: Case[] }>('hmm_viterbi');
const c = fixture.cases[0]!;

describe('hmmViterbi', () => {
  it('matches the plain-numpy max-sum recursion (PRML 13.68-13.69)', () => {
    const out = hmmViterbi(c.pi, c.A, c.B);
    out.omega.forEach((row, n) => row.forEach((v, k) => expect(v).toBeCloseTo(c.expected.omega[n]![k]!, 9)));
  });

  it('matches the plain-numpy backpointers', () => {
    const out = hmmViterbi(c.pi, c.A, c.B);
    // psi[0] has no predecessor and is not compared; the reference fills it with -1 as
    // a sentinel and the implementation is free to do the same or leave it unused.
    for (let n = 1; n < out.psi.length; n++) {
      expect(out.psi[n]).toEqual(c.expected.psi[n]);
    }
  });

  it('matches the plain-numpy backtracked path and log-probability (PRML 13.70-13.71)', () => {
    const out = hmmViterbi(c.pi, c.A, c.B);
    expect(out.path).toEqual(c.expected.path);
    expect(out.logProb).toBeCloseTo(c.expected.logProb, 9);
  });

  it('the single best path can disagree with the sequence of individually most probable states', () => {
    // PRML's point in 13.2.5: maximising gamma(zn) at every n separately need not
    // recover the Viterbi path. Found by search (see the golden fixture generator's
    // commit history) for a 2-state, 3-step chain and pinned here as a fixed regression.
    const pi: Vec = [0.69, 0.31];
    const A: Mat = [
      [0.94, 0.06],
      [0.42, 0.58],
    ];
    const B: Mat = [
      [0.19, 0.81],
      [0.74, 0.26],
      [0.15, 0.85],
    ];
    const { alphaHat, c: cScale } = hmmForwardScaled(pi, A, B);
    const betaHat = hmmBackwardScaled(A, B, cScale);
    const gamma = hmmGammaScaled(alphaHat, betaHat);
    const mostProbableIndividually = gamma.map((row) => row.indexOf(Math.max(...row)));
    expect(mostProbableIndividually).toEqual([1, 0, 1]);

    const viterbi = hmmViterbi(pi, A, B);
    expect(viterbi.path).toEqual([1, 1, 1]);
    expect(viterbi.path).not.toEqual(mostProbableIndividually);
  });
});
