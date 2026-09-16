import { useCanvasLayer, useFrame } from '../Plot.js';
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

const DEFAULT_MAX_LENGTH = 24;
const DEFAULT_HEAD_SIZE = 6;
const ARROWHEAD_SPREAD = Math.PI / 7;

export interface ScaledArrow {
  readonly originPx: readonly [number, number];
  readonly dxPx: number;
  readonly dyPx: number;
  /** Magnitude of `field(x, y)` in data units, for colouring by raw field strength. */
  readonly magnitude: number;
}

/**
 * Maps every origin through `field` and `toPx`, then rescales every arrow by one global
 * factor so the longest lands at exactly `maxLength` pixels on screen. Measuring the
 * longest arrow in pixel space (not data space) matters whenever `toPx` is anisotropic,
 * since a data-space vector along x and one along y of equal magnitude need not render
 * at equal pixel length.
 */
export function computeScaledArrows(
  origins: readonly (readonly [number, number])[],
  field: (x: number, y: number) => readonly [number, number],
  toPx: (x: number, y: number) => readonly [number, number],
  maxLength: number,
): ScaledArrow[] {
  const raw = origins.map(([x, y]) => {
    const [vx, vy] = field(x, y);
    const [ox, oy] = toPx(x, y);
    const [ex, ey] = toPx(x + vx, y + vy);
    return {
      originPx: [ox, oy] as const,
      dxPx: ex - ox,
      dyPx: ey - oy,
      magnitude: Math.hypot(vx, vy),
    };
  });

  const maxPixelMagnitude = raw.reduce((m, a) => Math.max(m, Math.hypot(a.dxPx, a.dyPx)), 0);
  const scale = maxPixelMagnitude > 0 ? maxLength / maxPixelMagnitude : 0;

  return raw.map((a) => ({
    originPx: a.originPx,
    dxPx: a.dxPx * scale,
    dyPx: a.dyPx * scale,
    magnitude: a.magnitude,
  }));
}

function drawArrow(ctx: CanvasRenderingContext2D, ox: number, oy: number, dx: number, dy: number, headSize: number): void {
  const ex = ox + dx;
  const ey = oy + dy;
  ctx.moveTo(ox, oy);
  ctx.lineTo(ex, ey);

  const length = Math.hypot(dx, dy);
  if (length === 0) return;
  const angle = Math.atan2(dy, dx);
  const size = Math.min(headSize, length);
  ctx.moveTo(ex - size * Math.cos(angle - ARROWHEAD_SPREAD), ey - size * Math.sin(angle - ARROWHEAD_SPREAD));
  ctx.lineTo(ex, ey);
  ctx.lineTo(ex - size * Math.cos(angle + ARROWHEAD_SPREAD), ey - size * Math.sin(angle + ARROWHEAD_SPREAD));
}

export function VectorField({
  origins,
  field,
  color,
  maxLength = DEFAULT_MAX_LENGTH,
  headSize = DEFAULT_HEAD_SIZE,
  opacity = 1,
  z = 0,
}: VectorFieldProps) {
  const frame = useFrame();

  useCanvasLayer(
    z,
    (ctx) => {
      const arrows = computeScaledArrows(origins, field, frame.toPx, maxLength);
      ctx.globalAlpha = opacity;
      ctx.lineCap = 'round';

      if (typeof color === 'string') {
        ctx.strokeStyle = color;
        ctx.beginPath();
        for (const arrow of arrows) {
          drawArrow(ctx, arrow.originPx[0], arrow.originPx[1], arrow.dxPx, arrow.dyPx, headSize);
        }
        ctx.stroke();
      } else {
        for (const arrow of arrows) {
          ctx.beginPath();
          ctx.strokeStyle = color(arrow.magnitude);
          drawArrow(ctx, arrow.originPx[0], arrow.originPx[1], arrow.dxPx, arrow.dyPx, headSize);
          ctx.stroke();
        }
      }
    },
    [origins, field, color, maxLength, headSize, opacity],
  );

  return null;
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

const DEFAULT_TRAJECTORY_WIDTH = 1.5;
const MIN_FADE_ALPHA = 0.15;

function alphaAt(index: number, count: number, fadeOlder: boolean): number {
  if (!fadeOlder || count <= 1) return 1;
  return MIN_FADE_ALPHA + (1 - MIN_FADE_ALPHA) * (index / (count - 1));
}

export function Trajectory({
  path,
  color,
  width = DEFAULT_TRAJECTORY_WIDTH,
  markers = false,
  fadeOlder = false,
  z = 0,
}: TrajectoryProps) {
  const frame = useFrame();

  useCanvasLayer(
    z,
    (ctx) => {
      if (path.length === 0) return;
      const pixels = path.map(([x, y]) => frame.toPx(x, y));

      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';

      for (let i = 1; i < pixels.length; i++) {
        const [x0, y0] = pixels[i - 1]!;
        const [x1, y1] = pixels[i]!;
        ctx.globalAlpha = alphaAt(i, pixels.length, fadeOlder);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }

      if (markers) {
        const radius = Math.max(1.5, width * 1.5);
        for (let i = 0; i < pixels.length; i++) {
          const [x, y] = pixels[i]!;
          ctx.globalAlpha = alphaAt(i, pixels.length, fadeOlder);
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    },
    [path, color, width, markers, fadeOlder],
  );

  return null;
}
