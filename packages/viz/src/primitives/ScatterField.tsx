import type { KeyboardEvent as ReactKeyboardEvent } from 'react';

import { useCanvasLayer, useFrame } from '../Plot.js';
import { useDrag, useResolvedTokens } from '../hooks.js';

export type PointShape = 'circle' | 'cross' | 'square' | 'triangle' | 'ring';

export interface PlotPoint {
  readonly x: number;
  readonly y: number;
  readonly color?: string;
  readonly shape?: PointShape;
  readonly size?: number;
  readonly opacity?: number;
  /** Stable across renders, so that dragging point 3 does not become dragging point 4 when the array is re-sorted. */
  readonly id?: string | number;
}

export interface ScatterFieldProps {
  points: readonly PlotPoint[];
  color?: string;
  shape?: PointShape;
  size?: number;
  opacity?: number;
  /** Enables dragging and reports the moved point by index. */
  onMove?: (index: number, x: number, y: number) => void;
  onHover?: (index: number | null) => void;
  onSelect?: (index: number) => void;
  label?: (point: PlotPoint, index: number) => string;
}

const DEFAULT_SIZE = 4.5;

function ShapeGlyph({ shape, size, color, opacity }: { shape: PointShape; size: number; color: string; opacity: number }) {
  switch (shape) {
    case 'square':
      return <rect x={-size} y={-size} width={size * 2} height={size * 2} fill={color} opacity={opacity} />;
    case 'cross':
      return (
        <path
          d={`M ${-size} 0 L ${size} 0 M 0 ${-size} L 0 ${size}`}
          stroke={color}
          strokeWidth={Math.max(1, size / 2.5)}
          opacity={opacity}
          fill="none"
        />
      );
    case 'triangle':
      return <polygon points={trianglePoints(size)} fill={color} opacity={opacity} />;
    case 'ring':
      return <circle r={size} fill="none" stroke={color} strokeWidth={Math.max(1, size / 3)} opacity={opacity} />;
    case 'circle':
    default:
      return <circle r={size} fill={color} opacity={opacity} />;
  }
}

function trianglePoints(size: number): string {
  const angles = [-Math.PI / 2, (Math.PI * 7) / 6, Math.PI / 6];
  return angles.map((a) => `${size * Math.cos(a)},${size * Math.sin(a)}`).join(' ');
}

interface ScatterPointProps {
  point: PlotPoint;
  index: number;
  color: string;
  shape: PointShape;
  size: number;
  opacity: number;
  onMove?: ((index: number, x: number, y: number) => void) | undefined;
  onHover?: ((index: number | null) => void) | undefined;
  onSelect?: ((index: number) => void) | undefined;
  label?: ((point: PlotPoint, index: number) => string) | undefined;
}

function ScatterPoint({ point, index, color, shape, size, opacity, onMove, onHover, onSelect, label }: ScatterPointProps) {
  const frame = useFrame();
  const [px, py] = frame.toPx(point.x, point.y);
  const drag = useDrag({ onDrag: (x, y) => onMove?.(index, x, y) });

  const interactive = Boolean(onMove || onSelect);
  const accessibleName = label ? label(point, index) : `Point ${index + 1}`;

  function handleKeyDown(event: ReactKeyboardEvent<SVGGElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      if (onSelect) {
        event.preventDefault();
        onSelect(index);
      }
      return;
    }
    if (!onMove) return;
    const step = event.shiftKey ? 10 : 1;
    let dx = 0;
    let dy = 0;
    if (event.key === 'ArrowLeft') dx = -step;
    else if (event.key === 'ArrowRight') dx = step;
    else if (event.key === 'ArrowUp') dy = -step;
    else if (event.key === 'ArrowDown') dy = step;
    else return;
    event.preventDefault();
    const [nx, ny] = frame.toData(px + dx, py + dy);
    onMove(index, nx, ny);
  }

  return (
    <g
      transform={`translate(${px}, ${py})`}
      role={onSelect ? 'button' : 'img'}
      aria-label={accessibleName}
      tabIndex={interactive ? 0 : -1}
      onKeyDown={interactive ? handleKeyDown : undefined}
      onClick={onSelect ? () => onSelect(index) : undefined}
      onPointerEnter={onHover ? () => onHover(index) : undefined}
      onPointerLeave={onHover ? () => onHover(null) : undefined}
      onPointerDown={onMove ? drag.onPointerDown : undefined}
      style={onMove ? drag.style : undefined}
    >
      <ShapeGlyph shape={shape} size={size} color={color} opacity={opacity} />
    </g>
  );
}

/**
 * Renders to SVG, not canvas, because points are the thing users grab and an SVG element
 * gets hit-testing, focus, and keyboard access for free. Beyond a few thousand points that
 * trade reverses; `ScatterCloud` covers the non-interactive bulk case.
 */
export function ScatterField({ points, color, shape, size, opacity, onMove, onHover, onSelect, label }: ScatterFieldProps) {
  const tokens = useResolvedTokens();

  return (
    <g>
      {points.map((point, index) => (
        <ScatterPoint
          key={point.id ?? index}
          point={point}
          index={index}
          color={point.color ?? color ?? tokens.color.accent}
          shape={point.shape ?? shape ?? 'circle'}
          size={point.size ?? size ?? DEFAULT_SIZE}
          opacity={point.opacity ?? opacity ?? 1}
          onMove={onMove}
          onHover={onHover}
          onSelect={onSelect}
          label={label}
        />
      ))}
    </g>
  );
}

export interface ScatterCloudProps {
  points: readonly (readonly [number, number])[];
  color: string;
  size?: number;
  opacity?: number;
  z?: number;
}

const DEFAULT_CLOUD_POINT_SIZE = 1.5;

/** Canvas-drawn, non-interactive points: MCMC samples, predictive draws, large datasets. */
export function ScatterCloud({ points, color, size = DEFAULT_CLOUD_POINT_SIZE, opacity = 1, z = 0 }: ScatterCloudProps) {
  const frame = useFrame();

  useCanvasLayer(
    z,
    (ctx) => {
      if (points.length === 0) return;
      const path = new Path2D();
      for (const [x, y] of points) {
        const [px, py] = frame.toPx(x, y);
        path.moveTo(px + size, py);
        path.arc(px, py, size, 0, Math.PI * 2);
      }
      ctx.fillStyle = color;
      ctx.globalAlpha = opacity;
      ctx.fill(path);
    },
    [points, color, size, opacity],
  );

  return null;
}
