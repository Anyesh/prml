import { NotImplemented } from '../frame.js';
import type { Interpolator } from './ColorScale.js';

export interface VectorFieldProps {
  /** Arrow origins in data coordinates. */
  origins: readonly (readonly [number, number])[];
  /** `(x, y)` to a vector in data units. */
  field: (x: number, y: number) => readonly [number, number];
  color: string | Interpolator;
  /**
   * Longest arrow in pixels. Vectors are scaled by a single global factor rather than
   * normalised individually, because relative magnitude is the point in a gradient field.
   */
  maxLength?: number;
  headSize?: number;
  opacity?: number;
  z?: number;
}

export function VectorField(props: VectorFieldProps) {
  void props;
  throw new NotImplemented('VectorField');
}

export interface TrajectoryProps {
  /** Ordered states, drawn as a connected path with the arrow of time shown by opacity. */
  path: readonly (readonly [number, number])[];
  color: string;
  width?: number;
  /** Draws a marker at each state, for MCMC where the count of stops carries information. */
  markers?: boolean;
  /** Fades the oldest states, so a long chain does not become an opaque blob. */
  fadeOlder?: boolean;
  z?: number;
}

export function Trajectory(props: TrajectoryProps) {
  void props;
  throw new NotImplemented('Trajectory');
}
