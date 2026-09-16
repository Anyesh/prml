import { useMemo } from 'react';
import { quantileSorted } from 'd3-array';
import { contours as createContours, type ContourMultiPolygon } from 'd3-contour';

import { useCanvasLayer, useFrame } from '../Plot.js';
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

const DEFAULT_LEVEL_COUNT = 10;
const DEFAULT_LINE_WIDTH = 1;

export interface FlattenedField {
  readonly values: readonly number[];
  readonly width: number;
  readonly height: number;
}

/**
 * Flattens `values[j][i]` into row-major order matching `d3-contour`'s expected layout
 * (width columns per row, `dy` rows). Keep `width`/`height` from this call rather than
 * re-deriving them elsewhere: a mismatched pair silently transposes the field.
 */
export function flattenField(data: FieldData): FlattenedField {
  const width = data.xs.length;
  const height = data.ys.length;
  const values = new Array<number>(width * height);
  for (let j = 0; j < height; j++) {
    const row = data.values[j]!;
    for (let i = 0; i < width; i++) {
      values[j * width + i] = row[i]!;
    }
  }
  return { values, width, height };
}

/**
 * Maps a fractional grid index into `coords`, linearly interpolating between neighbours
 * for a non-uniform grid and linearly extrapolating past the ends (contour rings can
 * extend half a cell beyond the outermost sample at the field's boundary).
 */
export function indexToCoord(t: number, coords: readonly number[]): number {
  const n = coords.length;
  if (n === 1) return coords[0]!;
  const i0 = Math.min(Math.max(Math.floor(t), 0), n - 2);
  const frac = t - i0;
  return coords[i0]! + frac * (coords[i0 + 1]! - coords[i0]!);
}

/**
 * `d3-contour` centers sample `i + j*width` at planar coordinate `(i + 0.5, j + 0.5)`
 * (documented in its README), not at `(i, j)`, so the 0.5 must come off before the
 * fractional index goes through `xs`/`ys`.
 */
export function contourPointToData(
  point: readonly number[],
  xs: readonly number[],
  ys: readonly number[],
): readonly [number, number] {
  return [indexToCoord(point[0]! - 0.5, xs), indexToCoord(point[1]! - 0.5, ys)];
}

/**
 * Explicit `levels` win outright. Otherwise takes `levelCount` interior quantiles of the
 * value distribution (not an evenly spaced range) so bands carry roughly equal mass on a
 * skewed field such as a posterior density.
 */
export function computeLevels(
  flatValues: readonly number[],
  levels: readonly number[] | undefined,
  levelCount: number = DEFAULT_LEVEL_COUNT,
): number[] {
  if (levels && levels.length > 0) return [...levels].sort((a, b) => a - b);

  const finite = flatValues.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (finite.length === 0) return [];

  const n = Math.max(1, Math.floor(levelCount));
  const out: number[] = [];
  for (let k = 1; k <= n; k++) {
    out.push(quantileSorted(finite, k / (n + 1))!);
  }
  return out;
}

function ringToPixels(
  ring: readonly (readonly number[])[],
  xs: readonly number[],
  ys: readonly number[],
  toPx: (x: number, y: number) => readonly [number, number],
): (readonly [number, number])[] {
  return ring.map((point) => {
    const [dataX, dataY] = contourPointToData(point, xs, ys);
    return toPx(dataX, dataY);
  });
}

function tracePolygon(
  ctx: CanvasRenderingContext2D,
  polygon: ContourMultiPolygon,
  xs: readonly number[],
  ys: readonly number[],
  toPx: (x: number, y: number) => readonly [number, number],
): void {
  for (const rings of polygon.coordinates) {
    for (const ring of rings) {
      const pixels = ringToPixels(ring, xs, ys, toPx);
      pixels.forEach(([px, py], i) => {
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
    }
  }
}

/**
 * Marching squares via `d3-contour`, painted to canvas. Contours are recomputed only when
 * `data` changes identity, so a widget dragging a slider must produce a new grid object
 * rather than mutating the old one in place.
 */
export function ContourField({
  data,
  levels,
  levelCount = DEFAULT_LEVEL_COUNT,
  color,
  fill,
  lineWidth = DEFAULT_LINE_WIDTH,
  opacity = 1,
  z = 0,
}: ContourFieldProps) {
  const frame = useFrame();

  const flattened = useMemo(() => flattenField(data), [data]);

  const resolvedLevels = useMemo(
    () => computeLevels(flattened.values, levels, levelCount),
    [flattened, levels, levelCount],
  );

  const contoursByLevel = useMemo(() => {
    if (flattened.width < 2 || flattened.height < 2 || resolvedLevels.length === 0) return [];
    const generator = createContours().size([flattened.width, flattened.height]);
    const flat = flattened.values as number[];
    return resolvedLevels.map((level) => ({ level, polygon: generator.contour(flat, level) }));
  }, [flattened, resolvedLevels]);

  useCanvasLayer(
    z,
    (ctx) => {
      const { xs, ys } = data;
      ctx.globalAlpha = opacity;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = 'round';

      if (fill) {
        for (const { level, polygon } of contoursByLevel) {
          ctx.beginPath();
          tracePolygon(ctx, polygon, xs, ys, frame.toPx);
          ctx.fillStyle = fill(level);
          ctx.fill();
        }
      }

      const strokeColorAt = typeof color === 'string' ? () => color : color;
      for (const { level, polygon } of contoursByLevel) {
        ctx.beginPath();
        tracePolygon(ctx, polygon, xs, ys, frame.toPx);
        ctx.strokeStyle = strokeColorAt(level);
        ctx.stroke();
      }
    },
    [contoursByLevel, color, fill, lineWidth, opacity],
  );

  return null;
}
