import { sigmoid, softmax } from '../numeric.js';
import type { Vec } from '../types.js';
import type { HiddenActivation, OutputActivation } from './types.js';

/** `h(a)`, PRML 5.58-5.59 for tanh; the logistic case is 5.6 reused as a hidden unit. */
export function applyHidden(kind: HiddenActivation, a: Vec): number[] {
  return kind === 'tanh' ? a.map(Math.tanh) : a.map(sigmoid);
}

/**
 * `h'(a)` expressed through `z = h(a)` rather than `a` directly, PRML 5.60 for tanh
 * (`1 - z^2`) and the analogous `z(1-z)` for the logistic unit: both forms are cheaper
 * and more stable than differentiating the raw activation again.
 */
export function hiddenDerivative(kind: HiddenActivation, z: number): number {
  return kind === 'tanh' ? 1 - z * z : z * (1 - z);
}

/** `h''(a)`, needed only by the exact second-derivative recursion in `hessian.ts`. */
export function hiddenSecondDerivative(kind: HiddenActivation, z: number): number {
  const hp = hiddenDerivative(kind, z);
  return kind === 'tanh' ? -2 * z * hp : hp * (1 - 2 * z);
}

/**
 * PRML 5.5 (logistic), 5.25 (softmax); linear output leaves `a` unchanged, which is
 * eq 5.9's regression case with the identity activation. Softmax is the one case that
 * mixes across units, so it cannot be written as a per-unit map like the other two.
 */
export function applyOutput(kind: OutputActivation, a: Vec): number[] {
  if (kind === 'linear') return [...a];
  if (kind === 'logistic') return a.map(sigmoid);
  return softmax(a);
}
