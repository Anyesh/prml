import type { Vec } from '../types.js';

/**
 * PRML 5.69, the central-difference estimate of a partial derivative, applied
 * component by component. Its error is `O(ε²)` against the forward difference's
 * `O(ε)` (5.68), which is what lets it stand in as an independent check on
 * backpropagation rather than only a cheap sanity test: at the step size below, its
 * own truncation error sits under the golden fixtures' 1e-9 tolerance.
 *
 * `eps` scales with `|x_i|` (the standard recipe for a central difference in double
 * precision) because a fixed step is too large relative to a tiny weight and too small
 * relative to a huge one; without scaling, cancellation error would dominate in the
 * first case and truncation error in the second.
 */
export function numericalGradient(f: (x: Vec) => number, x: Vec, epsScale = 1e-5): number[] {
  const grad = new Array<number>(x.length);
  for (let i = 0; i < x.length; i++) {
    const eps = epsScale * Math.max(1, Math.abs(x[i]!));
    const xPlus = x.slice();
    const xMinus = x.slice();
    xPlus[i] = x[i]! + eps;
    xMinus[i] = x[i]! - eps;
    grad[i] = (f(xPlus) - f(xMinus)) / (2 * eps);
  }
  return grad;
}
