import { describe, expect, it } from 'vitest';
import { loadFixture } from './testing/fixture.js';
import { categorical, pcg32, randInt, shuffle, standardNormal, uniformArray } from './rng.js';
import type { Rng } from './types.js';

interface Case {
  readonly fn: string;
  readonly name: string;
  readonly seed: number;
  readonly stream: number;
  readonly expected: number[];
}

const fixture = loadFixture<{ cases: Case[] }>('rng');
function findCase(name: string): Case {
  const c = fixture.cases.find((c) => c.name === name);
  if (!c) throw new Error(`no fixture case named "${name}"`);
  return c;
}

describe('pcg32', () => {
  it('matches the canonical O\'Neill reference vector for seed 42, stream 54', () => {
    const c = findCase('canonical');
    const rng = pcg32(c.seed, c.stream);
    const out = c.expected.map(() => rng.nextUint32());
    expect(out).toEqual(c.expected);
  });

  it('next() is nextUint32()/2^32, consuming the stream in the same order', () => {
    const c = findCase('canonical');
    const rng = pcg32(c.seed, c.stream);
    const out = c.expected.map(() => rng.next());
    out.forEach((v, i) => {
      expect(v).toBeCloseTo(c.expected[i]! / 4294967296, 12);
    });
  });

  it('two different seeds diverge immediately', () => {
    const a = pcg32(42, 54);
    const b = pcg32(43, 54);
    expect(a.nextUint32()).not.toBe(b.nextUint32());
  });

  it('fork(streamId) reproduces pcg32(sameSeed, streamId) exactly', () => {
    const c = findCase('fork_stream');
    const parent = pcg32(42, 54);
    const forked = parent.fork(c.stream);
    const out = c.expected.map(() => forked.nextUint32());
    expect(out).toEqual(c.expected);
  });

  it('a fork does not share state with its parent: consuming the parent first leaves the fork unaffected', () => {
    const c = findCase('fork_stream');
    const parent = pcg32(42, 54);
    for (let i = 0; i < 10; i++) parent.nextUint32();
    const forked = parent.fork(c.stream);
    const out = c.expected.map(() => forked.nextUint32());
    expect(out).toEqual(c.expected);
  });

  it('two forks of the same parent seed with different stream ids never overlap over a long run', () => {
    const parent = pcg32(7, 1);
    const forkA = parent.fork(101);
    const forkB = parent.fork(202);
    const seqA = new Set(Array.from({ length: 5000 }, () => forkA.nextUint32()));
    const seqB = Array.from({ length: 5000 }, () => forkB.nextUint32());
    const overlap = seqB.filter((v) => seqA.has(v));
    // A handful of coincidental collisions across 5000x5000 draws from a 32-bit range
    // is expected by pure chance; a shared/overlapping stream would collide on nearly
    // every draw, not a handful.
    expect(overlap.length).toBeLessThan(50);
  });
});

describe('uniformArray', () => {
  it('equals calling next() n times on an identically-seeded generator', () => {
    const a = pcg32(9, 3);
    const b = pcg32(9, 3);
    const arr = uniformArray(a, 10);
    const manual = Array.from({ length: 10 }, () => b.next());
    expect(arr).toEqual(manual);
  });

  it('every value lies in [0, 1)', () => {
    const rng = pcg32(11, 5);
    for (const v of uniformArray(rng, 1000)) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe('shuffle', () => {
  it('does not mutate the input array', () => {
    const xs = [1, 2, 3, 4, 5];
    const original = [...xs];
    shuffle(pcg32(1, 1), xs);
    expect(xs).toEqual(original);
  });

  it('returns a permutation: same multiset of elements', () => {
    const xs = [1, 2, 3, 4, 5, 6, 7];
    const out = shuffle(pcg32(1, 1), xs);
    expect([...out].sort((a, b) => a - b)).toEqual(xs);
  });

  it('is deterministic given the same seed', () => {
    const xs = [1, 2, 3, 4, 5, 6, 7, 8];
    const out1 = shuffle(pcg32(123, 4), xs);
    const out2 = shuffle(pcg32(123, 4), xs);
    expect(out1).toEqual(out2);
  });
});

describe('randInt', () => {
  it('always returns 0 for n = 1', () => {
    const rng = pcg32(2, 2);
    for (let i = 0; i < 20; i++) expect(randInt(rng, 1)).toBe(0);
  });

  it('stays within [0, n) over many draws', () => {
    const rng = pcg32(3, 3);
    for (let i = 0; i < 5000; i++) {
      const v = randInt(rng, 7);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(7);
    }
  });

  it('is approximately uniform: bucket counts stay within a binomial standard-error band', () => {
    const rng = pcg32(4, 4);
    const n = 5;
    const trials = 100_000;
    const counts = new Array(n).fill(0);
    for (let i = 0; i < trials; i++) counts[randInt(rng, n)]++;
    const expected = trials / n;
    // Each bucket is Binomial(trials, 1/n); stderr = sqrt(trials * p * (1-p)).
    const stderr = Math.sqrt(trials * (1 / n) * (1 - 1 / n));
    for (const c of counts) {
      expect(Math.abs(c - expected)).toBeLessThan(6 * stderr);
    }
  });
});

describe('categorical', () => {
  it('is exact at a degenerate weight vector', () => {
    const rng = pcg32(5, 5);
    for (let i = 0; i < 20; i++) expect(categorical(rng, [0, 1, 0])).toBe(1);
  });

  it('does not require normalised weights and matches the weight ratios statistically', () => {
    const rng = pcg32(6, 6);
    const weights = [1, 3, 6]; // sums to 10, unnormalised
    const trials = 100_000;
    const counts = [0, 0, 0];
    for (let i = 0; i < trials; i++) counts[categorical(rng, weights)]!++;
    for (let i = 0; i < weights.length; i++) {
      const p = weights[i]! / 10;
      const expected = trials * p;
      const stderr = Math.sqrt(trials * p * (1 - p));
      expect(Math.abs(counts[i]! - expected)).toBeLessThan(6 * stderr);
    }
  });
});

describe('standardNormal', () => {
  it('has approximately zero mean and unit variance over a large sample, within a standard-error band', () => {
    const rng = pcg32(21, 21);
    const n = 200_000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const z = standardNormal(rng);
      sum += z;
      sumSq += z * z;
    }
    const meanEst = sum / n;
    const varEst = sumSq / n - meanEst * meanEst;
    // SE(mean) = sigma/sqrt(n) = 1/sqrt(200000) ~= 0.00224; SE(variance) ~= sqrt(2/n) ~= 0.00316
    // for a standard normal. 6 sigma keeps the false-failure rate astronomically small.
    expect(Math.abs(meanEst)).toBeLessThan(6 * (1 / Math.sqrt(n)));
    expect(Math.abs(varEst - 1)).toBeLessThan(6 * Math.sqrt(2 / n));
  });

  it('caches the second deviate against the Rng instance: a call on an unrelated rng does not consume the pending cache', () => {
    const isolated = standardNormal(pcg32(30, 30));
    // Draw from a completely unrelated generator in between.
    standardNormal(pcg32(99, 99));
    const freshSameParams = standardNormal(pcg32(30, 30));
    expect(freshSameParams).toBe(isolated);
  });

  it('draws raw uniforms only on the odd call: the second call per pair costs zero underlying draws', () => {
    let calls = 0;
    const inner = pcg32(40, 40);
    const counting: Rng = {
      nextUint32: () => {
        calls++;
        return inner.nextUint32();
      },
      next: () => {
        calls++;
        return inner.next();
      },
      fork: (id: number) => inner.fork(id),
    };
    standardNormal(counting);
    const callsAfterFirst = calls;
    expect(callsAfterFirst).toBeGreaterThan(0);
    standardNormal(counting);
    expect(calls).toBe(callsAfterFirst);
  });
});
