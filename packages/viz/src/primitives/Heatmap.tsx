import { NotImplemented } from '../frame.js';
import type { Interpolator } from './ColorScale.js';
import type { FieldData } from './ContourField.js';

export interface HeatmapProps {
  data: FieldData;
  interpolator: Interpolator;
  opacity?: number;
  /**
   * Nearest-neighbour by default, so grid cells stay visibly discrete. Smoothing a
   * 40×40 evaluation up to 600px suggests a resolution the computation does not have.
   */
  smooth?: boolean;
  z?: number;
}

/**
 * Renders through an offscreen `ImageData` at grid resolution and then scales it, because
 * filling several thousand individual rectangles costs more per frame than the entire
 * density evaluation behind them.
 */
export function Heatmap(props: HeatmapProps) {
  void props;
  throw new NotImplemented('Heatmap');
}
