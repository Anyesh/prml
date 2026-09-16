import { describe, expect, it } from 'vitest';
import {
  clamp,
  evalGrid,
  linspace,
  logistic,
  logSumExp,
  mean,
  sigmoid,
  softmax,
  trapz,
  variance,
} from './numeric.js';

describe('logSumExp', () => {
  it('matches the direct sum for well-scaled inputs', () => {
    const xs = [1, 2, 3];
    const expected = Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3));
    expect(logSumExp(xs)).toBeCloseTo(expected, 12);
  });

  it('does not overflow for large inputs that would blow up a naive sum', () => {
    const xs = [1000, 1001, 1002];
    const expected = 1002 + Math.log(Math.exp(1000 - 1002) + Math.exp(1001 - 1002) + Math.exp(0));
    expect(logSumExp(xs)).toBeCloseTo(expected, 9);
    expect(Number.isFinite(logSumExp(xs))).toBe(true);
  });
});

describe('softmax', () => {
  it('sums to 1 and preserves relative ratios', () => {
    const xs = [1, 2, 3];
    const out = softmax(xs);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(out[1]! / out[0]!).toBeCloseTo(Math.exp(1), 9);
    expect(out[2]! / out[1]!).toBeCloseTo(Math.exp(1), 9);
  });

  it('is shift-invariant, matching a direct computation on shifted inputs', () => {
    const shifted = softmax([1001, 1002, 1003]);
    const base = softmax([1, 2, 3]);
    for (let i = 0; i < 3; i++) {
      expect(shifted[i]!).toBeCloseTo(base[i]!, 9);
    }
  });
});

describe('sigmoid / logistic', () => {
  it('agree with each other', () => {
    expect(sigmoid(0.37)).toBeCloseTo(logistic(0.37), 12);
  });

  it('matches the direct formula near zero', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5, 12);
    expect(sigmoid(1)).toBeCloseTo(1 / (1 + Math.exp(-1)), 12);
  });

  it('stays finite and saturates for large |x|', () => {
    expect(sigmoid(1000)).toBe(1);
    expect(sigmoid(-1000)).toBe(0);
  });
});

describe('linspace', () => {
  it('is inclusive of both endpoints', () => {
    const xs = linspace(0, 1, 5);
    expect(xs).toHaveLength(5);
    expect(xs[0]).toBe(0);
    expect(xs[4]).toBe(1);
    expect(xs[2]).toBeCloseTo(0.5, 12);
  });

  it('returns [a] for n = 1', () => {
    expect(linspace(3, 7, 1)).toEqual([3]);
  });
});

describe('clamp', () => {
  it('clamps into range and passes through inside it', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(11, 0, 10)).toBe(10);
  });
});

describe('mean / variance', () => {
  it('computes the exact mean and population variance for a known small set', () => {
    const xs = [2, 4, 4, 4, 5, 5, 7, 9];
    expect(mean(xs)).toBeCloseTo(5, 12);
    expect(variance(xs)).toBeCloseTo(4, 12);
  });
});

describe('trapz', () => {
  it('integrates a constant exactly', () => {
    const xs = [0, 1, 2, 3];
    const ys = [2, 2, 2, 2];
    expect(trapz(ys, xs)).toBeCloseTo(6, 12);
  });

  it('integrates a linear ramp exactly on an unequally spaced grid', () => {
    const xs = [0, 1, 3];
    const ys = xs.map((x) => 2 * x + 1);
    // Antiderivative of 2x+1 is x^2+x; trapz is exact for a linear integrand
    // regardless of spacing, so this closed form is the correct check, not a magic number.
    const antiderivative = (x: number) => x * x + x;
    const expected = antiderivative(3) - antiderivative(0);
    expect(trapz(ys, xs)).toBeCloseTo(expected, 9);
  });
});

describe('evalGrid', () => {
  it('uses row-major values[j][i] = f(xs[i], ys[j]) on a non-square grid', () => {
    const xs = [0, 1, 2];
    const ys = [10, 20];
    const grid = evalGrid(xs, ys, (x, y) => x + y);
    expect(grid.xs).toEqual(xs);
    expect(grid.ys).toEqual(ys);
    expect(grid.values).toHaveLength(2);
    expect(grid.values[0]).toHaveLength(3);
    expect(grid.values[0]).toEqual([10, 11, 12]);
    expect(grid.values[1]).toEqual([20, 21, 22]);
  });
});
