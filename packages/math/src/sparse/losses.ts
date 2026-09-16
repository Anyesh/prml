/**
 * The margin-based error functions of Figure 7.5, each written as a function of the single
 * quantity `z = t * y(x)`: positive and growing means a confident correct prediction,
 * negative means a confident mistake. Writing all four this way is what makes them
 * comparable on one axis.
 */

/** PRML 7.45: the hinge loss `[1 - z]+`. Exactly zero for any correctly classified point past the margin. */
export function hingeLoss(margin: number): number {
  return Math.max(0, 1 - margin);
}

/**
 * PRML 7.48 rescaled by `1/ln 2` (Figure 7.5's caption) so it passes through `(0, 1)` like
 * the hinge does, making the two directly comparable on the same axes.
 */
export function logisticMarginLoss(margin: number): number {
  return Math.log1p(Math.exp(-margin)) / Math.LN2;
}

/**
 * Squared error re-expressed in terms of the margin. For `t in {-1, 1}`, `(y - t)^2`
 * expands to `(z - 1)^2` once `y = zt` and `t^2 = 1` are substituted, which is what makes
 * it grow for confidently correct points instead of flattening out like the other three.
 */
export function squaredMarginLoss(margin: number): number {
  const d = 1 - margin;
  return d * d;
}

/** The 0/1 misclassification error as a function of the margin: 1 for any wrong or boundary prediction. */
export function misclassificationLoss(margin: number): number {
  return margin > 0 ? 0 : 1;
}

/** PRML 7.51: zero inside `|residual| < epsilon`, linear beyond it. */
export function epsilonInsensitiveLoss(residual: number, epsilon: number): number {
  const abs = Math.abs(residual);
  return abs < epsilon ? 0 : abs - epsilon;
}
