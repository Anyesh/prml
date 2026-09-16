import { NotImplemented } from '../frame.js';
import type { Interpolator } from './ColorScale.js';

export interface FieldData {
  readonly xs: readonly number[];
  readonly ys: readonly number[];
  /** `values[j][i]` at `(xs[i], ys[j])`, matching `evalGrid` in `@prml/math`. */
  readonly values: readonly (readonly number[])[];
}

export interface ContourFieldProps {
  data: FieldData;
  /** Explicit contour levels. Omit to take `levelCount` quantiles of the value range. */
  levels?: readonly number[];
  levelCount?: number;
  /** Line colour, or an interpolator to colour each level by its value. */
  color: string | Interpolator;
  fill?: Interpolator;
  lineWidth?: number;
  opacity?: number;
  z?: number;
}

/**
 * Marching squares via `d3-contour`, painted to canvas. Contours are recomputed only when
 * `data` changes identity, so a widget dragging a slider must produce a new grid object
 * rather than mutating the old one in place.
 */
export function ContourField(props: ContourFieldProps) {
  void props;
  throw new NotImplemented('ContourField');
}
