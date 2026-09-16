import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  hmmBackwardScaled,
  hmmBackwardUnscaled,
  hmmForwardScaled,
  hmmForwardUnscaled,
  hmmGammaScaled,
  hmmGammaUnscaled,
  hmmLikelihoodUnscaled,
  hmmLogLikelihoodScaled,
  hmmXiScaled,
  hmmXiUnscaled,
} from './forwardBackward.js';
import type { Mat, Vec } from '../types.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('hmm_forward_backward');
function caseFor(fn: string): Case {
  const c = fixture.cases.find((c) => c.fn === fn);
  if (!c) throw new Error(`no case for ${fn}`);
  return c;
}

function expectMatClose(actual: Mat, expected: Mat, digits = 9) {
  actual.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, digits)));
}

describe('hmmForwardUnscaled', () => {
  it('matches the plain-numpy alpha recursion (PRML 13.36-13.37)', () => {
    const c = caseFor('hmmForwardUnscaled');
    const out = hmmForwardUnscaled(c['pi'] as Vec, c['A'] as Mat, c['B'] as Mat);
    expectMatClose(out, c['expected'] as Mat);
  });
});

describe('hmmBackwardUnscaled', () => {
  it('matches the plain-numpy beta recursion (PRML 13.38)', () => {
    const c = caseFor('hmmBackwardUnscaled');
    const out = hmmBackwardUnscaled(c['A'] as Mat, c['B'] as Mat);
    expectMatClose(out, c['expected'] as Mat);
  });
});

describe('hmmGammaUnscaled', () => {
  it('matches alpha*beta normalised per time step (PRML 13.33)', () => {
    const c = caseFor('hmmGammaUnscaled');
    const out = hmmGammaUnscaled(c['alpha'] as Mat, c['beta'] as Mat);
    expectMatClose(out, c['expected'] as Mat);
  });

  it('every row sums to one', () => {
    const c = caseFor('hmmGammaUnscaled');
    const out = hmmGammaUnscaled(c['alpha'] as Mat, c['beta'] as Mat);
    for (const row of out) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });
});

describe('hmmLikelihoodUnscaled', () => {
  it('matches sum_zN alpha(zN) (PRML 13.42)', () => {
    const c = caseFor('hmmLikelihoodUnscaled');
    expect(hmmLikelihoodUnscaled(c['alpha'] as Mat)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('hmmXiUnscaled', () => {
  it('matches the plain-numpy pairwise posterior (PRML 13.43)', () => {
    const c = caseFor('hmmXiUnscaled');
    const out = hmmXiUnscaled(c['alpha'] as Mat, c['A'] as Mat, c['B'] as Mat, c['beta'] as Mat, c['pX'] as number);
    const expected = c['expected'] as Mat[];
    out.forEach((mat, n) => expectMatClose(mat, expected[n]!));
  });
});

describe('hmmForwardScaled', () => {
  it('matches the plain-numpy scaled alpha recursion and scaling factors (PRML 13.59)', () => {
    const c = caseFor('hmmForwardScaled');
    const { alphaHat, c: cOut } = hmmForwardScaled(c['pi'] as Vec, c['A'] as Mat, c['B'] as Mat);
    const expected = c['expected'] as { alphaHat: Mat; c: Vec };
    expectMatClose(alphaHat, expected.alphaHat);
    cOut.forEach((v, n) => expect(v).toBeCloseTo(expected.c[n]!, 9));
  });

  it('every alphaHat row sums to one', () => {
    const c = caseFor('hmmForwardScaled');
    const { alphaHat } = hmmForwardScaled(c['pi'] as Vec, c['A'] as Mat, c['B'] as Mat);
    for (const row of alphaHat) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });
});

describe('hmmBackwardScaled', () => {
  it('matches the plain-numpy scaled beta recursion (PRML 13.62)', () => {
    const c = caseFor('hmmBackwardScaled');
    const out = hmmBackwardScaled(c['A'] as Mat, c['B'] as Mat, c['c'] as Vec);
    expectMatClose(out, c['expected'] as Mat);
  });
});

describe('hmmGammaScaled', () => {
  it('matches alphaHat * betaHat directly, no extra normalisation (PRML 13.64)', () => {
    const c = caseFor('hmmGammaScaled');
    const out = hmmGammaScaled(c['alphaHat'] as Mat, c['betaHat'] as Mat);
    expectMatClose(out, c['expected'] as Mat);
  });
});

describe('hmmXiScaled', () => {
  it('matches the plain-numpy scaled pairwise posterior (PRML 13.65)', () => {
    const c = caseFor('hmmXiScaled');
    const out = hmmXiScaled(c['alphaHat'] as Mat, c['A'] as Mat, c['B'] as Mat, c['betaHat'] as Mat, c['c'] as Vec);
    const expected = c['expected'] as Mat[];
    out.forEach((mat, n) => expectMatClose(mat, expected[n]!));
  });
});

describe('hmmLogLikelihoodScaled', () => {
  it('matches sum(log(c)) (PRML 13.63)', () => {
    const c = caseFor('hmmLogLikelihoodScaled');
    expect(hmmLogLikelihoodScaled(c['c'] as Vec)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('scaled vs unscaled agreement on a short chain', () => {
  it('gamma and xi from the scaled recursion equal the unscaled recursion exactly, before underflow', () => {
    const fb = caseFor('hmmForwardUnscaled');
    const pi = fb['pi'] as Vec;
    const A = fb['A'] as Mat;
    const B = fb['B'] as Mat;

    const alpha = hmmForwardUnscaled(pi, A, B);
    const beta = hmmBackwardUnscaled(A, B);
    const gammaUnscaled = hmmGammaUnscaled(alpha, beta);
    const pX = hmmLikelihoodUnscaled(alpha);
    const xiUnscaled = hmmXiUnscaled(alpha, A, B, beta, pX);

    const { alphaHat, c } = hmmForwardScaled(pi, A, B);
    const betaHat = hmmBackwardScaled(A, B, c);
    const gammaScaled = hmmGammaScaled(alphaHat, betaHat);
    const xiScaled = hmmXiScaled(alphaHat, A, B, betaHat, c);

    // 1e-9 rather than exact equality: both paths are floating point, and this test's
    // point is that no *normalisation* mistake sneaks in, not that operation order is
    // bit-identical.
    expectMatClose(gammaScaled, gammaUnscaled);
    xiScaled.forEach((mat, n) => expectMatClose(mat, xiUnscaled[n]!));
  });
});

describe('why scaling exists', () => {
  it('the unscaled alpha recursion underflows to exactly zero on a long, uninformative chain', () => {
    const K = 2;
    const N = 200;
    const pi: Vec = [0.5, 0.5];
    const A: Mat = [
      [0.5, 0.5],
      [0.5, 0.5],
    ];
    // A weak, ambiguous emission at every step: no single observation is strongly
    // informative, so nothing here is different from a poorly-instrumented sensor, and
    // alpha's mass shrinks by roughly the same tiny factor at every one of 200 steps.
    const B: Mat = Array.from({ length: N }, () => [0.02, 0.018]);
    const alpha = hmmForwardUnscaled(pi, A, B);
    expect(alpha[N - 1]!.every((v) => v === 0)).toBe(true);

    const { c } = hmmForwardScaled(pi, A, B);
    const logLikelihood = hmmLogLikelihoodScaled(c);
    expect(Number.isFinite(logLikelihood)).toBe(true);
    expect(K).toBe(2);
  });
});
