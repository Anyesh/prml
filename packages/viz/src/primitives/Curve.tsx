import { NotImplemented } from '../frame.js';

export type Dash = 'solid' | 'dashed' | 'dotted';

export interface CurveProps {
  points: readonly (readonly [number, number])[];
  /** A resolved colour from `useResolvedTokens`, never a literal. */
  color: string;
  width?: number;
  dash?: Dash;
  opacity?: number;
  z?: number;
  /**
   * Monotone interpolation through the points. Off by default because a sampled density
   * curve must show its own sampling, and smoothing it hides undersampling from the reader.
   */
  smooth?: boolean;
}

export function Curve(props: CurveProps) {
  void props;
  throw new NotImplemented('Curve');
}

export interface BandProps {
  /** `[x, lower, upper]` per sample, in increasing `x`. */
  points: readonly (readonly [number, number, number])[];
  color: string;
  opacity?: number;
  z?: number;
}

/** The filled region between two curves: predictive variance ribbons and credible intervals. */
export function Band(props: BandProps) {
  void props;
  throw new NotImplemented('Band');
}

export interface FunctionCurveProps extends Omit<CurveProps, 'points'> {
  f: (x: number) => number;
  /** Defaults to the frame's x domain. */
  domain?: readonly [number, number];
  samples?: number;
}

/**
 * Samples `f` across the domain at render time. Separate from `Curve` because the sample
 * count must follow the pixel width, and a caller that precomputed points cannot know it.
 */
export function FunctionCurve(props: FunctionCurveProps) {
  void props;
  throw new NotImplemented('FunctionCurve');
}
