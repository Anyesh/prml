/** PRML 14.20's summand form, exp(-z), z = t * f(x). */
export function exponentialMarginError(z: number): number {
  return Math.exp(-z);
}

/** ln(1 + exp(-z)), same softplus identity as `classification/irls.ts`'s local helper. */
function softplus(a: number): number {
  return Math.max(a, 0) + Math.log1p(Math.exp(-Math.abs(a)));
}

/** ln(1 + exp(-z)), the z = t*y form of the cross-entropy error (PRML 4.90 rewritten for t in {-1,1}, discussed in 14.3.2). */
export function crossEntropyMarginError(z: number): number {
  return softplus(-z);
}

/** max(0, 1 - z), the SVM hinge error, for the figure-14.3 comparison. */
export function hingeMarginError(z: number): number {
  return Math.max(0, 1 - z);
}

/** 1 if z < 0 else 0, the ideal misclassification error. */
export function misclassificationMarginError(z: number): number {
  return z < 0 ? 1 : 0;
}

/** Regression residual error for figure 14.4. */
export function squaredResidualError(residual: number): number {
  return residual * residual;
}

/** Regression residual error for figure 14.4. */
export function absoluteResidualError(residual: number): number {
  return Math.abs(residual);
}
