import type { Vec } from '../types.js';
import type { KernelFunction } from './functions.js';

/** PRML 6.13: scaling a valid kernel by a positive constant leaves it valid. */
export function scaleKernel(k: KernelFunction, c: number): KernelFunction {
  return (x, xp) => c * k(x, xp);
}

/** PRML 6.17: the sum of two valid kernels is valid. */
export function sumKernel(a: KernelFunction, b: KernelFunction): KernelFunction {
  return (x, xp) => a(x, xp) + b(x, xp);
}

/** PRML 6.18: the product of two valid kernels is valid. */
export function productKernel(a: KernelFunction, b: KernelFunction): KernelFunction {
  return (x, xp) => a(x, xp) * b(x, xp);
}

/** PRML 6.6: the Gram matrix, `Knm = k(xn, xm)`. */
export function gramMatrix(kernel: KernelFunction, xs: readonly Vec[]): number[][] {
  return xs.map((xn) => xs.map((xm) => kernel(xn, xm)));
}

/** The vector `k(x)` of PRML 6.9, the kernel evaluated between every training point and one query. */
export function kernelVector(kernel: KernelFunction, trainX: readonly Vec[], x: Vec): number[] {
  return trainX.map((xn) => kernel(xn, x));
}
