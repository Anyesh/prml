import { normalPdf, normalSample, pcg32 } from '@prml/math';

/**
 * One fixed synthetic dataset shared by every 2.5 figure, so bin width, bandwidth, and K
 * are compared on identical data rather than each drawing its own sample.
 */
export const COMPONENTS = [
  { mu: -1.5, sigma2: 0.25 },
  { mu: 1.5, sigma2: 0.3 },
] as const;

export function truePdf(x: number): number {
  return 0.5 * normalPdf(x, COMPONENTS[0]) + 0.5 * normalPdf(x, COMPONENTS[1]);
}

export const DOMAIN_LO = -4;
export const DOMAIN_HI = 4;

export const DATA: readonly number[] = (() => {
  const rng = pcg32(20260916, 5);
  const points: number[] = [];
  for (let i = 0; i < 50; i++) {
    const c = rng.next() < 0.5 ? COMPONENTS[0] : COMPONENTS[1];
    points.push(normalSample(rng, c));
  }
  return points.sort((a, b) => a - b);
})();
