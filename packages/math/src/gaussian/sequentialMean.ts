/**
 * The sequential maximum-likelihood mean update, PRML 2.126: `mu_ML^(N) = mu_ML^(N-1) +
 * (1/N)(x_N - mu_ML^(N-1))`.
 *
 * PRML derives this as the Robbins-Monro root-finding update (2.129-2.136) applied to
 * the ML stationarity condition, not as an algebraic rearrangement of the batch mean, so
 * it generalises to settings with no closed-form batch estimate at all. `n` is the
 * count including the new observation `x`, i.e. this is the `N`-th point seen.
 */
export function sequentialMean(previous: number, n: number, x: number): number {
  return previous + (x - previous) / n;
}
