import { useRef } from 'react';
import { extent } from 'd3-array';

import { useCanvasLayer, useFrame } from '../Plot.js';
import { cssColorToRgbaBytes, type Interpolator } from './ColorScale.js';
import { flattenField, type FieldData } from './ContourField.js';

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

const LUT_SIZE = 256;

/**
 * Samples `interpolator` at `size` evenly spaced values across `domain` and parses each
 * result once, so drawing a grid of thousands of cells does one colour parse per bucket
 * rather than one per pixel.
 */
export function buildColorLut(
  interpolator: Interpolator,
  domain: readonly [number, number],
  size: number = LUT_SIZE,
  toRgba: (css: string) => readonly [number, number, number, number] = cssColorToRgbaBytes,
): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(size * 4);
  const [lo, hi] = domain;
  const span = hi - lo;
  for (let k = 0; k < size; k++) {
    const t = size === 1 || span === 0 ? lo : lo + (k / (size - 1)) * span;
    const [r, g, b, a] = toRgba(interpolator(t));
    lut[k * 4] = r;
    lut[k * 4 + 1] = g;
    lut[k * 4 + 2] = b;
    lut[k * 4 + 3] = a;
  }
  return lut;
}

/** Nearest lookup-table bucket for `value` given the domain the table was built over. */
export function lutIndexForValue(value: number, domain: readonly [number, number], size: number): number {
  if (!Number.isFinite(value)) return 0;
  const [lo, hi] = domain;
  if (hi === lo) return 0;
  const t = (value - lo) / (hi - lo);
  return Math.min(size - 1, Math.max(0, Math.round(t * (size - 1))));
}

type OffscreenCtx = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

interface OffscreenEntry {
  readonly width: number;
  readonly height: number;
  readonly canvas: OffscreenCanvas | HTMLCanvasElement;
  readonly ctx: OffscreenCtx;
  /** Identity of the `data`/`interpolator` pair this canvas's pixels were painted from. */
  readonly data: FieldData;
  readonly interpolator: Interpolator;
}

function createOffscreen(width: number, height: number, data: FieldData, interpolator: Interpolator): OffscreenEntry {
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('@prml/viz: 2D context unavailable for Heatmap offscreen canvas');
    return { width, height, canvas, ctx, data, interpolator };
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('@prml/viz: 2D context unavailable for Heatmap offscreen canvas');
  return { width, height, canvas, ctx, data, interpolator };
}

/**
 * Renders through an offscreen `ImageData` at grid resolution and then scales it, because
 * filling several thousand individual rectangles costs more per frame than the entire
 * density evaluation behind them.
 */
export function Heatmap({ data, interpolator, opacity = 1, smooth = false, z = 0 }: HeatmapProps) {
  const frame = useFrame();
  const offscreenRef = useRef<OffscreenEntry | null>(null);

  useCanvasLayer(
    z,
    (ctx) => {
      const { xs, ys } = data;
      const width = xs.length;
      const height = ys.length;
      if (width < 1 || height < 1) return;

      const cached = offscreenRef.current;
      const reusable =
        cached &&
        cached.width === width &&
        cached.height === height &&
        cached.data === data &&
        cached.interpolator === interpolator;

      const offscreen = reusable ? cached : createOffscreen(width, height, data, interpolator);
      offscreenRef.current = offscreen;

      if (!reusable) {
        const flattened = flattenField(data);
        const [lo, hi] = extent(flattened.values.filter(Number.isFinite));
        const domain: readonly [number, number] = [lo ?? 0, hi ?? 1];
        const lut = buildColorLut(interpolator, domain);

        const image = offscreen.ctx.createImageData(width, height);
        for (let j = 0; j < height; j++) {
          const row = data.values[j]!;
          const destRow = height - 1 - j;
          for (let i = 0; i < width; i++) {
            const idx = lutIndexForValue(row[i]!, domain, LUT_SIZE);
            const destIndex = (destRow * width + i) * 4;
            image.data[destIndex] = lut[idx * 4]!;
            image.data[destIndex + 1] = lut[idx * 4 + 1]!;
            image.data[destIndex + 2] = lut[idx * 4 + 2]!;
            image.data[destIndex + 3] = lut[idx * 4 + 3]!;
          }
        }
        offscreen.ctx.putImageData(image, 0, 0);
      }

      const [x0, y0] = frame.toPx(xs[0]!, ys[0]!);
      const [x1, y1] = frame.toPx(xs[width - 1]!, ys[height - 1]!);
      const destX = Math.min(x0, x1);
      const destY = Math.min(y0, y1);
      const destW = Math.abs(x1 - x0);
      const destH = Math.abs(y1 - y0);

      ctx.save();
      ctx.imageSmoothingEnabled = smooth;
      ctx.globalAlpha = opacity;
      ctx.drawImage(offscreen.canvas as CanvasImageSource, 0, 0, width, height, destX, destY, destW, destH);
      ctx.restore();
    },
    [data, interpolator, opacity, smooth],
  );

  return null;
}
