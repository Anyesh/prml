import { useId, useLayoutEffect, useRef, useState } from 'react';
import { scaleLinear } from 'd3-scale';

import { useFrame } from '../Plot.js';
import { useResolvedTokens } from '../hooks.js';
import { quantize, withAlpha, type Interpolator } from './ColorScale.js';

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

const OUTER_MARGIN = 8;
const PADDING = 8;
const ROW_HEIGHT = 14;
const ROW_GAP = 6;
const COLUMN_GAP = 16;
const MARK_WIDTH = 18;
const MARK_LABEL_GAP = 6;

const FONT_STYLE = {
  fontFamily: 'var(--prml-font-sans)',
  fontSize: 'var(--prml-text-xs)',
} as const;

function EntryMark({ entry }: { entry: LegendEntry }) {
  switch (entry.mark ?? 'line') {
    case 'dot':
      return <circle cx={MARK_WIDTH / 2} cy={0} r={4} fill={entry.color} />;
    case 'swatch':
      return <rect x={0} y={-5} width={MARK_WIDTH} height={10} fill={entry.color} />;
    case 'dashed-line':
      return <line x1={0} x2={MARK_WIDTH} y1={0} y2={0} stroke={entry.color} strokeWidth={2} strokeDasharray={`${MARK_WIDTH / 3} ${MARK_WIDTH / 6}`} />;
    case 'line':
    default:
      return <line x1={0} x2={MARK_WIDTH} y1={0} y2={0} stroke={entry.color} strokeWidth={2} />;
  }
}

function placementOrigin(
  placement: NonNullable<LegendProps['placement']>,
  innerWidth: number,
  innerHeight: number,
  boxWidth: number,
  boxHeight: number,
): readonly [number, number] {
  switch (placement) {
    case 'top-left':
      return [OUTER_MARGIN, OUTER_MARGIN];
    case 'bottom-left':
      return [OUTER_MARGIN, innerHeight - boxHeight - OUTER_MARGIN];
    case 'bottom-right':
      return [innerWidth - boxWidth - OUTER_MARGIN, innerHeight - boxHeight - OUTER_MARGIN];
    case 'top-right':
    default:
      return [innerWidth - boxWidth - OUTER_MARGIN, OUTER_MARGIN];
  }
}

/**
 * Positioned inside the plot area per `placement`, with a translucent backing so it stays
 * readable over a dense field underneath.
 */
export function Legend({ entries, placement = 'top-right', orientation = 'vertical' }: LegendProps) {
  const frame = useFrame();
  const tokens = useResolvedTokens();
  const rowRefs = useRef<Array<SVGGElement | null>>([]);
  const [rowWidths, setRowWidths] = useState<number[]>([]);

  useLayoutEffect(() => {
    setRowWidths(rowRefs.current.map((node) => (node ? node.getBBox().width : 0)));
  }, [entries]);

  if (entries.length === 0) return null;

  const positions: Array<{ x: number; y: number }> = [];
  let cursor = 0;
  for (let i = 0; i < entries.length; i++) {
    const width = rowWidths[i] ?? 0;
    if (orientation === 'horizontal') {
      positions.push({ x: cursor, y: 0 });
      cursor += width + COLUMN_GAP;
    } else {
      positions.push({ x: 0, y: cursor });
      cursor += ROW_HEIGHT + ROW_GAP;
    }
  }

  const contentWidth = orientation === 'horizontal' ? Math.max(0, cursor - COLUMN_GAP) : Math.max(0, ...rowWidths);
  const contentHeight = orientation === 'horizontal' ? ROW_HEIGHT : Math.max(0, cursor - ROW_GAP);
  const boxWidth = contentWidth + PADDING * 2;
  const boxHeight = contentHeight + PADDING * 2;

  const [ox, oy] = placementOrigin(placement, frame.innerWidth, frame.innerHeight, boxWidth, boxHeight);

  return (
    <g transform={`translate(${Math.round(ox)}, ${Math.round(oy)})`} aria-hidden="true">
      <rect width={boxWidth} height={boxHeight} rx={4} fill={withAlpha(tokens.color.surface, 0.85)} stroke={tokens.color.border} />
      <g transform={`translate(${PADDING}, ${PADDING + ROW_HEIGHT / 2})`}>
        {entries.map((entry, i) => {
          const pos = positions[i] ?? { x: 0, y: 0 };
          return (
            <g
              key={entry.label}
              ref={(node) => {
                rowRefs.current[i] = node;
              }}
              transform={`translate(${pos.x}, ${pos.y})`}
            >
              <EntryMark entry={entry} />
              <text x={MARK_WIDTH + MARK_LABEL_GAP} y={0} dominantBaseline="middle" fill={tokens.color.ink} style={FONT_STYLE}>
                {entry.label}
              </text>
            </g>
          );
        })}
      </g>
    </g>
  );
}

export interface ColorBarProps {
  interpolator: Interpolator;
  domain: readonly [number, number];
  label?: string;
  ticks?: number;
}

const BAR_WIDTH = 14;
const BAR_LENGTH = 120;
const GRADIENT_STOPS = 12;
const TICK_LENGTH = 4;
const TICK_LABEL_GAP = 4;

/**
 * A gradient strip with ticks, built from `quantize`d stops as SVG `<stop>` elements.
 * Renders at the local origin: `ColorBarProps` carries no placement, so a caller composes
 * it into a figure with its own wrapping `<g transform="translate(...)">`.
 */
export function ColorBar({ interpolator, domain, label, ticks = 5 }: ColorBarProps) {
  const tokens = useResolvedTokens();
  const gradientId = useId();

  const stops = quantize(interpolator, GRADIENT_STOPS);
  const axisScale = scaleLinear().domain(domain).range([BAR_LENGTH, 0]);
  const tickValues = axisScale.ticks(ticks);
  const format = axisScale.tickFormat(ticks);

  const oy = label ? 16 : 0;

  return (
    <g transform={`translate(0, ${oy})`} aria-hidden="true">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="1" x2="0" y2="0">
          {stops.map((color, i) => (
            <stop key={i} offset={`${(i / (stops.length - 1)) * 100}%`} stopColor={color} />
          ))}
        </linearGradient>
      </defs>
      <rect width={BAR_WIDTH} height={BAR_LENGTH} fill={`url(#${gradientId})`} stroke={tokens.color.border} />
      {tickValues.map((t) => {
        const y = Math.round(axisScale(t)) + 0.5;
        return (
          <g key={t}>
            <line x1={BAR_WIDTH} x2={BAR_WIDTH + TICK_LENGTH} y1={y} y2={y} stroke={tokens.color.axisLine} strokeWidth={1} />
            <text x={Math.round(BAR_WIDTH + TICK_LENGTH + TICK_LABEL_GAP)} y={y} dominantBaseline="middle" fill={tokens.color.inkMuted} style={FONT_STYLE}>
              {format(t)}
            </text>
          </g>
        );
      })}
      {label && (
        <text x={Math.round(BAR_WIDTH / 2)} y={-8} textAnchor="middle" fill={tokens.color.inkMuted} style={FONT_STYLE}>
          {label}
        </text>
      )}
    </g>
  );
}
