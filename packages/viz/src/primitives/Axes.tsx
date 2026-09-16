import { NotImplemented } from '../frame.js';

export interface AxisSpec {
  /** Omit to let d3 choose; pass explicit values where the book's figure uses particular ticks. */
  ticks?: readonly number[];
  tickCount?: number;
  label?: string;
  format?: (value: number) => string;
  /** Hides the line and ticks but keeps the label, for the inset panels that share an axis. */
  bare?: boolean;
}

export interface AxesProps {
  x?: AxisSpec | false;
  y?: AxisSpec | false;
  grid?: boolean;
  /** Draws a line at the given data value on each axis, for decision boundaries at zero. */
  zeroLine?: boolean;
}

/**
 * Axes render to SVG rather than canvas because tick text at 11px must hit the pixel grid
 * to stay legible, and canvas text does not hint.
 */
export function Axes(props: AxesProps) {
  void props;
  throw new NotImplemented('Axes');
}
