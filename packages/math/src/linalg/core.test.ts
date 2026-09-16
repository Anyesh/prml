import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  diag,
  diagOf,
  dot,
  eye,
  isSymmetric,
  matAdd,
  matmul,
  matScale,
  matSub,
  matvec,
  matZeros,
  norm,
  outer,
  quadForm,
  subvector,
  submatrix,
  symmetrise,
  trace,
  transpose,
  vecAdd,
  vecScale,
  vecSub,
  zeros,
} from './core.js';

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

describe('constructors', () => {
  it('zeros(n) is a length-n array of zeros', () => {
    expect(zeros(4)).toEqual([0, 0, 0, 0]);
  });

  it('matZeros(rows, cols) is a rows x cols array of zeros', () => {
    expect(matZeros(2, 3)).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it('eye(n) is the identity, eye(n, scale) scales the diagonal', () => {
    expect(eye(3)).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    expect(eye(2, 5)).toEqual([
      [5, 0],
      [0, 5],
    ]);
  });

  it('diag and diagOf are inverses', () => {
    const c = findCase('diag');
    const values = c['values'] as number[];
    const expected = c['expected'] as number[][];
    expect(diag(values)).toEqual(expected);
    expect(diagOf(expected)).toEqual(values);
  });
});

describe('golden-checked arithmetic', () => {
  it('matmul', () => {
    const c = findCase('matmul');
    const out = matmul(c['a'] as number[][], c['b'] as number[][]);
    const expected = c['expected'] as number[][];
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('matvec', () => {
    const c = findCase('matvec');
    const out = matvec(c['a'] as number[][], c['x'] as number[]);
    const expected = c['expected'] as number[];
    out.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
  });

  it('quadForm', () => {
    const c = findCase('quadForm');
    const out = quadForm(c['x'] as number[], c['a'] as number[][], c['y'] as number[]);
    expect(out).toBeCloseTo(c['expected'] as number, 9);
  });

  it('matAdd / matSub / matScale', () => {
    const add = findCase('matAdd');
    const sub = findCase('matSub');
    const scale = findCase('matScale');
    expect(matAdd(add['a'] as number[][], add['b'] as number[][])).toEqual(add['expected']);
    expect(matSub(sub['a'] as number[][], sub['b'] as number[][])).toEqual(sub['expected']);
    const scaled = matScale(scale['a'] as number[][], scale['s'] as number);
    const expected = scale['expected'] as number[][];
    scaled.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });

  it('vecAdd / vecSub / vecScale', () => {
    const add = findCase('vecAdd');
    const sub = findCase('vecSub');
    const scale = findCase('vecScale');
    expect(vecAdd(add['a'] as number[], add['b'] as number[])).toEqual(add['expected']);
    expect(vecSub(sub['a'] as number[], sub['b'] as number[])).toEqual(sub['expected']);
    const scaled = vecScale(scale['a'] as number[], scale['s'] as number);
    const expected = scale['expected'] as number[];
    scaled.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
  });

  it('dot, outer, norm, trace', () => {
    const dc = findCase('dot');
    const oc = findCase('outer');
    const nc = findCase('norm');
    const tc = findCase('trace');
    expect(dot(dc['a'] as number[], dc['b'] as number[])).toBeCloseTo(dc['expected'] as number, 9);
    const o = outer(oc['a'] as number[], oc['b'] as number[]);
    const oe = oc['expected'] as number[][];
    o.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(oe[i]![j]!, 9)));
    expect(norm(nc['a'] as number[])).toBeCloseTo(nc['expected'] as number, 9);
    expect(trace(tc['a'] as number[][])).toBeCloseTo(tc['expected'] as number, 9);
  });

  it('transpose', () => {
    const c = findCase('transpose');
    expect(transpose(c['a'] as number[][])).toEqual(c['expected']);
  });

  it('submatrix / subvector', () => {
    const sm = findCase('submatrix');
    const sv = findCase('subvector');
    expect(
      submatrix(sm['a'] as number[][], sm['rows'] as number[], sm['cols'] as number[]),
    ).toEqual(sm['expected']);
    expect(subvector(sv['x'] as number[], sv['idx'] as number[])).toEqual(sv['expected']);
  });

  it('symmetrise averages a matrix with its transpose', () => {
    const c = findCase('symmetrise');
    const out = symmetrise(c['a'] as number[][]);
    const expected = c['expected'] as number[][];
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('isSymmetric', () => {
  it('is true for a symmetric matrix and false for an asymmetric one', () => {
    expect(
      isSymmetric([
        [1, 2],
        [2, 1],
      ]),
    ).toBe(true);
    expect(
      isSymmetric([
        [1, 2],
        [3, 1],
      ]),
    ).toBe(false);
  });

  it('tolerates asymmetry within tol but not beyond it', () => {
    const nearlySymmetric = [
      [1, 2],
      [2 + 1e-12, 1],
    ];
    expect(isSymmetric(nearlySymmetric, 1e-9)).toBe(true);
    expect(isSymmetric(nearlySymmetric, 1e-14)).toBe(false);
  });
});
