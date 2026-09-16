import { useFrame } from '../Plot.js';
import { useResolvedTokens } from '../hooks.js';

export type Anchor = 'start' | 'middle' | 'end';

export interface AnnotationProps {
  /** Data coordinates of the point being labelled. */
  x: number;
  y: number;
  text: string;
  color?: string;
  anchor?: Anchor;
  /** Pixel nudge, applied after projection, to clear the mark being labelled. */
  dx?: number;
  dy?: number;
  /**
   * Draws a translucent plate behind the text. Needed over a contour or heatmap, where
   * bare text is unreadable against half the colour ramp.
   */
  plate?: boolean;
  size?: 'xs' | 'sm';
}

/**
 * A short label anchored to a point in data space.
 *
 * SVG rather than canvas for the same reason as `Axes`: small text has to hit the pixel
 * grid to stay legible, and canvas text is unhinted. Callers that would otherwise push a
 * number into a caption should use this instead, because a value written beside the thing
 * it measures is read, and one written below the plot is not.
 */
export function Annotation({
  x,
  y,
  text,
  color,
  anchor = 'middle',
  dx = 0,
  dy = 0,
  plate = false,
  size = 'xs',
}: AnnotationProps) {
  const frame = useFrame();
  const tokens = useResolvedTokens();
  const [px, py] = frame.toPx(x, y);

  return (
    <g transform={`translate(${px + dx}, ${py + dy})`} style={{ pointerEvents: 'none' }}>
      {plate ? (
        <text
          textAnchor={anchor}
          dominantBaseline="middle"
          stroke={tokens.color.plotBg}
          strokeWidth={3.5}
          strokeLinejoin="round"
          fill="none"
          style={{ font: `var(--prml-text-${size}) var(--prml-font-sans)` }}
        >
          {text}
        </text>
      ) : null}
      <text
        textAnchor={anchor}
        dominantBaseline="middle"
        fill={color ?? tokens.color.inkMuted}
        style={{ font: `var(--prml-text-${size}) var(--prml-font-sans)` }}
      >
        {text}
      </text>
    </g>
  );
}

export interface RuleProps {
  /** Exactly one of these: a horizontal rule at `y`, or a vertical rule at `x`. */
  x?: number;
  y?: number;
  color?: string;
  dash?: boolean;
  width?: number;
  label?: string;
}

/**
 * A reference line across the plot at a fixed data value: a threshold, a true value, a
 * precision limit. Distinct from `Axes`'s `zeroLine`, which only ever draws at zero.
 */
export function Rule({ x, y, color, dash = true, width = 1, label }: RuleProps) {
  const frame = useFrame();
  const tokens = useResolvedTokens();
  const stroke = color ?? tokens.color.borderStrong;

  if ((x === undefined) === (y === undefined)) {
    throw new Error('@prml/viz: Rule takes exactly one of x or y');
  }

  const horizontal = y !== undefined;
  const [px, py] = frame.toPx(x ?? 0, y ?? 0);

  return (
    <g style={{ pointerEvents: 'none' }}>
      <line
        x1={horizontal ? 0 : px}
        x2={horizontal ? frame.innerWidth : px}
        y1={horizontal ? py : 0}
        y2={horizontal ? py : frame.innerHeight}
        stroke={stroke}
        strokeWidth={width}
        strokeDasharray={dash ? '4 3' : undefined}
      />
      {label ? (
        <text
          x={horizontal ? frame.innerWidth - 4 : px + 4}
          y={horizontal ? py - 4 : 10}
          textAnchor={horizontal ? 'end' : 'start'}
          fill={stroke}
          style={{ font: 'var(--prml-text-xs) var(--prml-font-sans)' }}
        >
          {label}
        </text>
      ) : null}
    </g>
  );
}
