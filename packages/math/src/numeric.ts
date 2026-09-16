import type { Vec } from './types.js';

/**
 * Must be computed by shifting on the maximum, so that responsibilities in a mixture
 * with well-separated components do not underflow to a zero denominator.
 */
export function logSumExp(xs: Vec): number {
  if (xs.length === 0) return -Infinity;
  let max = -Infinity;
  for (const x of xs) if (x > max) max = x;
  if (!Number.isFinite(max)) return max;
  let sum = 0;
  for (const x of xs) sum += Math.exp(x - max);
  return max + Math.log(sum);
}

export function softmax(xs: Vec): number[] {
  const lse = logSumExp(xs);
  return xs.map((x) => Math.exp(x - lse));
}

/** Must stay finite for |x| beyond 700, where a naive `1/(1+exp(-x))` overflows. */
export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export function logistic(x: number): number {
  return sigmoid(x);
}

/** Inclusive of both endpoints; `n` of 1 returns `[a]`. */
export function linspace(a: number, b: number, n: number): number[] {
  if (n === 1) return [a];
  const step = (b - a) / (n - 1);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = a + step * i;
  out[n - 1] = b;
  return out;
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

export function mean(xs: Vec): number {
  let sum = 0;
  for (const x of xs) sum += x;
  return sum / xs.length;
}

/** Population variance (divides by `n`), matching numpy's default rather than R's. */
export function variance(xs: Vec): number {
  const m = mean(xs);
  let sum = 0;
  for (const x of xs) sum += (x - m) * (x - m);
  return sum / xs.length;
}

/** Trapezoidal rule over unequally spaced `xs`; `xs` must be strictly increasing. */
export function trapz(ys: Vec, xs: Vec): number {
  let sum = 0;
  for (let i = 1; i < xs.length; i++) {
    const dx = xs[i]! - xs[i - 1]!;
    sum += (dx * (ys[i]! + ys[i - 1]!)) / 2;
  }
  return sum;
}

export interface Grid2D {
  readonly xs: number[];
  readonly ys: number[];
  /** `values[j][i]` is the function at `(xs[i], ys[j])`, matching row-major raster order. */
  readonly values: number[][];
}

/**
 * Evaluates `f` on the outer product of `xs` and `ys`. The row-major convention here is
 * what `ContourField` and `Heatmap` in `@prml/viz` consume directly, so changing it
 * breaks every field widget.
 */
export function evalGrid(
  xs: Vec,
  ys: Vec,
  f: (x: number, y: number) => number,
): Grid2D {
  const values: number[][] = new Array(ys.length);
  for (let j = 0; j < ys.length; j++) {
    const row = new Array<number>(xs.length);
    for (let i = 0; i < xs.length; i++) {
      row[i] = f(xs[i]!, ys[j]!);
    }
    values[j] = row;
  }
  return { xs: [...xs], ys: [...ys], values };
}
