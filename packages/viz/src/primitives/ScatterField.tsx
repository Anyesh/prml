import { NotImplemented } from '../frame.js';

export type PointShape = 'circle' | 'cross' | 'square' | 'triangle' | 'ring';

export interface PlotPoint {
  readonly x: number;
  readonly y: number;
  readonly color?: string;
  readonly shape?: PointShape;
  readonly size?: number;
  readonly opacity?: number;
  /** Stable across renders, so that dragging point 3 does not become dragging point 4 when the array is re-sorted. */
  readonly id?: string | number;
}

export interface ScatterFieldProps {
  points: readonly PlotPoint[];
  color?: string;
  shape?: PointShape;
  size?: number;
  opacity?: number;
  /** Enables dragging and reports the moved point by index. */
  onMove?: (index: number, x: number, y: number) => void;
  onHover?: (index: number | null) => void;
  onSelect?: (index: number) => void;
  label?: (point: PlotPoint, index: number) => string;
}

/**
 * Renders to SVG, not canvas, because points are the thing users grab and an SVG element
 * gets hit-testing, focus, and keyboard access for free. Beyond a few thousand points that
 * trade reverses; `ScatterCloud` covers the non-interactive bulk case.
 */
export function ScatterField(props: ScatterFieldProps) {
  void props;
  throw new NotImplemented('ScatterField');
}

export interface ScatterCloudProps {
  points: readonly (readonly [number, number])[];
  color: string;
  size?: number;
  opacity?: number;
  z?: number;
}

/** Canvas-drawn, non-interactive points: MCMC samples, predictive draws, large datasets. */
export function ScatterCloud(props: ScatterCloudProps) {
  void props;
  throw new NotImplemented('ScatterCloud');
}
