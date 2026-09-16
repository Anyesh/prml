import type { Frame } from '../frame.js';
import { useCanvasLayer } from '../Plot.js';

export type BarOrientation = 'vertical' | 'horizontal';

export interface BarDatum {
  /** Position along the category axis, in data coordinates. Integer slots read best. */
  at: number;
  value: number;
  /** A resolved colour from `useResolvedTokens`, never a literal. */
  color: string;
  opacity?: number;
}

export interface BarsProps {
  bars: readonly BarDatum[];
  /**
   * Thickness along the category axis, in data units. The default leaves a visible gap
   * between adjacent integer slots, which is what separates a bar chart from a histogram.
   */
  thickness?: number;
  baseline?: number;
  /** 'horizontal' puts the category on the y axis, for labels too long to fit under a bar. */
  orientation?: BarOrientation;
  z?: number;
}

export interface PixelRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/**
 * Bars are anchored at the baseline rather than at the value, so a negative value grows
 * downwards from it instead of inverting the rectangle into a zero-height sliver.
 */
export function barRect(
  frame: Frame,
  bar: BarDatum,
  thickness: number,
  baseline: number,
  orientation: BarOrientation,
): PixelRect {
  const half = thickness / 2;

  if (orientation === 'vertical') {
    const [left] = frame.toPx(bar.at - half, 0);
    const [right] = frame.toPx(bar.at + half, 0);
    const [, valuePx] = frame.toPx(0, bar.value);
    const [, basePx] = frame.toPx(0, baseline);
    return {
      x: Math.min(left, right),
      y: Math.min(valuePx, basePx),
      width: Math.abs(right - left),
      height: Math.abs(basePx - valuePx),
    };
  }

  const [, top] = frame.toPx(0, bar.at + half);
  const [, bottom] = frame.toPx(0, bar.at - half);
  const [valuePx] = frame.toPx(bar.value, 0);
  const [basePx] = frame.toPx(baseline, 0);
  return {
    x: Math.min(valuePx, basePx),
    y: Math.min(top, bottom),
    width: Math.abs(basePx - valuePx),
    height: Math.abs(bottom - top),
  };
}

/**
 * Rectangles from a baseline to a value, one per category.
 *
 * Distinct from `Heatmap`, which fills a grid, and from `Curve`, which joins samples of a
 * continuous function: a bar chart's category axis has no distance between neighbours, so
 * interpolating across it would assert something untrue. Pair it with `Axes` given explicit
 * `ticks` and a `format` that maps the slot index to its label.
 */
export function Bars({
  bars,
  thickness = 0.7,
  baseline = 0,
  orientation = 'vertical',
  z = 0,
}: BarsProps) {
  useCanvasLayer(
    z,
    (ctx, frame) => {
      ctx.save();
      for (const bar of bars) {
        if (!Number.isFinite(bar.at) || !Number.isFinite(bar.value)) continue;
        const rect = barRect(frame, bar, thickness, baseline, orientation);
        ctx.globalAlpha = bar.opacity ?? 1;
        ctx.fillStyle = bar.color;
        ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
      }
      ctx.restore();
    },
    [bars, thickness, baseline, orientation, z],
  );

  return null;
}
