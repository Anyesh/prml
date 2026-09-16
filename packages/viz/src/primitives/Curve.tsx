import { curveLinear, curveMonotoneX, line as shapeLine } from 'd3-shape';

import type { Frame } from '../frame.js';
import { useCanvasLayer, useFrame } from '../Plot.js';

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

function applyDash(ctx: CanvasRenderingContext2D, dash: Dash | undefined, width: number): void {
  switch (dash) {
    case 'dashed':
      ctx.lineCap = 'butt';
      ctx.setLineDash([width * 3, width * 2]);
      break;
    case 'dotted':
      // A zero-length dash with a round cap draws a dot instead of a segment.
      ctx.lineCap = 'round';
      ctx.setLineDash([0, width * 2.5]);
      break;
    default:
      ctx.lineCap = 'butt';
      ctx.setLineDash([]);
  }
}

function strokePoints(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  points: readonly (readonly [number, number])[],
  smooth: boolean,
): void {
  const generator = shapeLine<readonly [number, number]>()
    .defined(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
    .x(([x, y]) => frame.toPx(x, y)[0])
    .y(([x, y]) => frame.toPx(x, y)[1])
    .curve(smooth ? curveMonotoneX : curveLinear)
    .context(ctx);
  ctx.beginPath();
  generator(points);
  ctx.stroke();
}

export function Curve({ points, color, width = 1.5, dash, opacity = 1, z = 0, smooth = false }: CurveProps) {
  useCanvasLayer(
    z,
    (ctx, frame) => {
      if (points.length < 2) return;
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      applyDash(ctx, dash, width);
      strokePoints(ctx, frame, points, smooth);
      ctx.restore();
    },
    [points, color, width, dash, opacity, z, smooth],
  );

  return null;
}

export interface BandProps {
  /** `[x, lower, upper]` per sample, in increasing `x`. */
  points: readonly (readonly [number, number, number])[];
  color: string;
  opacity?: number;
  z?: number;
}

function segmentBand(
  points: readonly (readonly [number, number, number])[],
): Array<Array<readonly [number, number, number]>> {
  const segments: Array<Array<readonly [number, number, number]>> = [];
  let current: Array<readonly [number, number, number]> = [];
  for (const p of points) {
    const [x, lo, hi] = p;
    if (Number.isFinite(x) && Number.isFinite(lo) && Number.isFinite(hi)) {
      current.push(p);
    } else if (current.length) {
      segments.push(current);
      current = [];
    }
  }
  if (current.length) segments.push(current);
  return segments;
}

/** The filled region between two curves: predictive variance ribbons and credible intervals. */
export function Band({ points, color, opacity = 0.2, z = -1 }: BandProps) {
  useCanvasLayer(
    z,
    (ctx, frame) => {
      const segments = segmentBand(points);
      if (!segments.length) return;
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.fillStyle = color;
      for (const segment of segments) {
        if (segment.length < 2) continue;
        ctx.beginPath();
        segment.forEach(([x, , hi], i) => {
          const [px, py] = frame.toPx(x, hi);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        for (const [x, lo] of [...segment].reverse()) {
          const [px, py] = frame.toPx(x, lo);
          ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    },
    [points, color, opacity, z],
  );

  return null;
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
export function FunctionCurve({
  f,
  domain,
  samples,
  color,
  width = 1.5,
  dash,
  opacity = 1,
  z = 0,
  smooth = false,
}: FunctionCurveProps) {
  const frame = useFrame();
  const [d0, d1] = domain ?? frame.xScale.domain();

  useCanvasLayer(
    z,
    (ctx, innerFrame) => {
      // Roughly one sample per 1-2 device pixels, since the caller cannot know the
      // rendered width when it builds the widget.
      const n = samples ?? Math.max(2, Math.ceil((Math.max(1, innerFrame.innerWidth) * innerFrame.dpr) / 1.5));
      const points: Array<readonly [number, number]> = Array.from({ length: n }, (_, i) => {
        const x = d0 + ((d1 - d0) * i) / (n - 1);
        return [x, f(x)] as const;
      });

      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      applyDash(ctx, dash, width);
      strokePoints(ctx, innerFrame, points, smooth);
      ctx.restore();
    },
    [f, d0, d1, samples, color, width, dash, opacity, z, smooth],
  );

  return null;
}
