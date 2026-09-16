import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { eye, matmul, transpose } from './core.js';
import {
  cholesky,
  det,
  eigSym,
  inverse,
  jitter,
  logDet,
  pinv,
  solve,
  solveCholesky,
  solveMat,
  svd,
} from './decompose.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('linalg');
function findCase(name: string): Case {
  const c = fixture.cases.find((c) => c.fn === name);
  if (!c) throw new Error(`no fixture case named "${name}"`);
  return c;
}

function maxAbsDiff(a: number[][], b: number[][]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i]!.length; j++) {
      m = Math.max(m, Math.abs(a[i]![j]! - b[i]![j]!));
    }
  }
  return m;
}

describe('cholesky', () => {
  it('matches numpy for a known SPD matrix', () => {
    const c = findCase('cholesky');
    const L = cholesky(c['a'] as number[][]);
    const expected = c['expected'] as number[][];
    L.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('throws, rather than returning NaN, for a non-positive-definite symmetric input', () => {
    const notPD = [
      [1, 2],
      [2, 1],
    ];
    expect(() => cholesky(notPD)).toThrow();
    let threw = false;
    try {
      cholesky(notPD);
    } catch (e) {
      threw = true;
      expect(Number.isNaN(e as unknown as number)).toBe(false);
    }
    expect(threw).toBe(true);
  });

  it('throws for a non-symmetric input', () => {
    expect(() =>
      cholesky([
        [1, 2],
        [3, 4],
      ]),
    ).toThrow();
  });
});

describe('jitter', () => {
  it('adds eps to the diagonal only', () => {
    const a = [
      [1, 2],
      [2, 3],
    ];
    const out = jitter(a, 0.01);
    expect(out[0]![0]).toBeCloseTo(1.01, 12);
    expect(out[1]![1]).toBeCloseTo(3.01, 12);
    expect(out[0]![1]).toBe(2);
    expect(out[1]![0]).toBe(2);
  });

  it('defaults eps to 1e-8', () => {
    const a = [[5]];
    expect(jitter(a)[0]![0]).toBeCloseTo(5 + 1e-8, 15);
  });
});

describe('logDet', () => {
  it('matches numpy slogdet, computed via cholesky rather than log(det)', () => {
    const c = findCase('logDet');
    expect(logDet(c['a'] as number[][])).toBeCloseTo(c['expected'] as number, 9);
  });

  it('stays finite where Math.log(det(A)) would underflow to -Infinity', () => {
    const n = 60;
    const a: number[][] = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 0.1 : 0)),
    );
    // det(A) = 0.1^60, which underflows Math.log(det(A)) to -Infinity; logDet must not.
    expect(Number.isFinite(logDet(a))).toBe(true);
    expect(logDet(a)).toBeCloseTo(n * Math.log(0.1), 6);
  });
});

describe('det / inverse / solve / solveMat on a general invertible matrix', () => {
  it('det', () => {
    const c = findCase('det');
    expect(det(c['a'] as number[][])).toBeCloseTo(c['expected'] as number, 9);
  });

  it('inverse', () => {
    const c = findCase('inverse');
    const out = inverse(c['a'] as number[][]);
    const expected = c['expected'] as number[][];
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('solve', () => {
    const c = findCase('solve');
    const out = solve(c['a'] as number[][], c['b'] as number[]);
    const expected = c['expected'] as number[];
    out.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
  });

  it('solveMat', () => {
    const c = findCase('solveMat');
    const out = solveMat(c['a'] as number[][], c['b'] as number[][]);
    const expected = c['expected'] as number[][];
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('solveCholesky', () => {
  it('solves Ax=b given L=cholesky(A), matching numpy.linalg.solve(A, b)', () => {
    const c = findCase('solveCholesky');
    const out = solveCholesky(c['l'] as number[][], c['b'] as number[]);
    const expected = c['expected'] as number[];
    out.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
  });
});

describe('eigSym', () => {
  it('returns eigenvalues descending and sign-fixed eigenvectors as rows, matching numpy eigh (transposed, reversed, sign-fixed)', () => {
    const c = findCase('eigSym');
    const { values, vectors } = eigSym(c['a'] as number[][]);
    const expected = c['expected'] as { values: number[]; vectors: number[][] };
    const expectedValues = expected.values;
    const expectedVectors = expected.vectors;
    values.forEach((v, i) => expect(v).toBeCloseTo(expectedValues[i]!, 9));
    vectors.forEach((row, i) =>
      row.forEach((v, j) => expect(v).toBeCloseTo(expectedVectors[i]![j]!, 9)),
    );
  });

  it('sign convention is deterministic: the largest-magnitude entry of each eigenvector is positive', () => {
    const c = findCase('eigSym');
    const { vectors } = eigSym(c['a'] as number[][]);
    for (const row of vectors) {
      let maxIdx = 0;
      for (let i = 1; i < row.length; i++) {
        if (Math.abs(row[i]!) > Math.abs(row[maxIdx]!)) maxIdx = i;
      }
      expect(row[maxIdx]!).toBeGreaterThan(0);
    }
  });

  it('descending order: values[0] >= values[1] >= values[2]', () => {
    const c = findCase('eigSym');
    const { values } = eigSym(c['a'] as number[][]);
    expect(values[0]!).toBeGreaterThanOrEqual(values[1]!);
    expect(values[1]!).toBeGreaterThanOrEqual(values[2]!);
  });
});

describe('svd', () => {
  it('singular values match numpy, descending', () => {
    const c = findCase('svd');
    const { s } = svd(c['a'] as number[][]);
    const expected = (c['expected'] as { s: number[] }).s;
    s.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
    expect(s[0]!).toBeGreaterThanOrEqual(s[1]!);
    expect(s[1]!).toBeGreaterThanOrEqual(s[2]!);
  });

  it('reconstructs the original matrix: U diag(s) V^T = A (no sign convention is documented for U/V, so only the round trip is golden-comparable)', () => {
    const c = findCase('svd');
    const a = c['a'] as number[][];
    const { u, s, v } = svd(a);
    const sMat = s.map((sv, i) => s.map((_, j) => (i === j ? sv : 0)));
    const reconstructed = matmul(matmul(u, sMat), transpose(v));
    expect(maxAbsDiff(reconstructed, a as number[][])).toBeLessThan(1e-9);
  });

  it('U and V have orthonormal columns', () => {
    const c = findCase('svd');
    const { u, v } = svd(c['a'] as number[][]);
    const utu = matmul(transpose(u), u);
    const vtv = matmul(transpose(v), v);
    expect(maxAbsDiff(utu, eye(utu.length))).toBeLessThan(1e-9);
    expect(maxAbsDiff(vtv, eye(vtv.length))).toBeLessThan(1e-9);
  });
});

describe('pinv', () => {
  it('matches numpy.linalg.pinv (the Moore-Penrose inverse is sign-of-U/V independent, so this is directly comparable)', () => {
    const c = findCase('pinv');
    const out = pinv(c['a'] as number[][]);
    const expected = c['expected'] as number[][];
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('satisfies the Moore-Penrose property A pinv(A) A = A', () => {
    const c = findCase('pinv');
    const a = c['a'] as number[][];
    const p = pinv(a);
    const roundTrip = matmul(matmul(a, p), a);
    expect(maxAbsDiff(roundTrip, a)).toBeLessThan(1e-8);
  });
});
