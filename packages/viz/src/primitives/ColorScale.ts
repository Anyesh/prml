import { NotImplemented } from '../frame.js';

export type Interpolator = (t: number) => string;

/**
 * Piecewise-linear interpolation through `stops` in Oklab rather than sRGB, because
 * interpolating viridis in sRGB reintroduces the luminance banding it was designed to
 * remove, and banding in a density field reads as structure that is not there.
 */
export function interpolateStops(stops: readonly string[]): Interpolator {
  void stops;
  throw new NotImplemented('interpolateStops');
}

export function sequentialScale(
  domain: readonly [number, number],
  stops?: readonly string[],
): Interpolator {
  void domain;
  void stops;
  throw new NotImplemented('sequentialScale');
}

/**
 * Symmetric about `center`, so that equal positive and negative values get equally
 * saturated colours no matter how lopsided the data range is.
 */
export function divergingScale(
  domain: readonly [number, number],
  center?: number,
  stops?: readonly string[],
): Interpolator {
  void domain;
  void center;
  void stops;
  throw new NotImplemented('divergingScale');
}

/** `n` evenly spaced samples, for discrete legends and contour band fills. */
export function quantize(interpolator: Interpolator, n: number): string[] {
  void interpolator;
  void n;
  throw new NotImplemented('quantize');
}

/**
 * Must accept every colour form the tokens can resolve to, including hex and `oklch()`,
 * because a caller cannot know which one a given theme produced.
 */
export function withAlpha(color: string, alpha: number): string {
  void color;
  void alpha;
  throw new NotImplemented('withAlpha');
}
