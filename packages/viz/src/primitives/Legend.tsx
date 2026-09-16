import { NotImplemented } from '../frame.js';
import type { Interpolator } from './ColorScale.js';

export interface LegendEntry {
  readonly label: string;
  readonly color: string;
  readonly mark?: 'line' | 'dashed-line' | 'dot' | 'swatch';
}

export interface LegendProps {
  entries: readonly LegendEntry[];
  placement?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  orientation?: 'vertical' | 'horizontal';
}

export function Legend(props: LegendProps) {
  void props;
  throw new NotImplemented('Legend');
}

export interface ColorBarProps {
  interpolator: Interpolator;
  domain: readonly [number, number];
  label?: string;
  ticks?: number;
}

export function ColorBar(props: ColorBarProps) {
  void props;
  throw new NotImplemented('ColorBar');
}
