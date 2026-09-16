import { useFrame } from '../Plot.js';
import { useResolvedTokens } from '../hooks.js';

export interface AxisSpec {
  /** Omit to let d3 choose; pass explicit values where the book's figure uses particular ticks. */
  ticks?: readonly number[];
  tickCount?: number;
  label?: string;
  format?: (value: number) => string;
  /** Hides the line and ticks but keeps the label, for the inset panels that share an axis. */
  bare?: boolean;
}

export interface AxesProps {
  x?: AxisSpec | false;
  y?: AxisSpec | false;
  grid?: boolean;
  /** Draws a line at the given data value on each axis, for decision boundaries at zero. */
  zeroLine?: boolean;
}

const DEFAULT_TICK_COUNT = 6;
const TICK_LENGTH = 5;
const LABEL_GAP = 4;
const AXIS_LABEL_GAP = 18;

const FONT_STYLE = {
  fontFamily: 'var(--prml-font-sans)',
  fontSize: 'var(--prml-text-xs)',
} as const;

function resolveSpec(spec: AxisSpec | false | undefined): AxisSpec | null {
  return spec === false ? null : (spec ?? {});
}

/**
 * Axes render to SVG rather than canvas because tick text at 11px must hit the pixel grid
 * to stay legible, and canvas text does not hint.
 */
export function Axes({ x, y, grid = false, zeroLine = false }: AxesProps) {
  const frame = useFrame();
  const tokens = useResolvedTokens();
  const { innerWidth, innerHeight, xScale, yScale } = frame;

  const xSpec = resolveSpec(x);
  const ySpec = resolveSpec(y);

  const xTickCount = xSpec?.tickCount ?? DEFAULT_TICK_COUNT;
  const yTickCount = ySpec?.tickCount ?? DEFAULT_TICK_COUNT;
  const xTicks = xSpec ? (xSpec.ticks ?? xScale.ticks(xTickCount)) : [];
  const yTicks = ySpec ? (ySpec.ticks ?? yScale.ticks(yTickCount)) : [];
  const xFormat = xSpec?.format ?? xScale.tickFormat(xTickCount);
  const yFormat = ySpec?.format ?? yScale.tickFormat(yTickCount);

  const xDomain = xScale.domain();
  const yDomain = yScale.domain();
  const showXZero = zeroLine && 0 >= Math.min(...xDomain) && 0 <= Math.max(...xDomain);
  const showYZero = zeroLine && 0 >= Math.min(...yDomain) && 0 <= Math.max(...yDomain);

  const axisLineY = Math.round(innerHeight) + 0.5;

  return (
    <g aria-hidden="true">
      {grid &&
        xSpec &&
        xTicks.map((t) => {
          const px = Math.round(xScale(t)) + 0.5;
          return (
            <line key={`grid-x-${t}`} x1={px} x2={px} y1={0} y2={innerHeight} stroke={tokens.color.gridLine} strokeWidth={1} />
          );
        })}
      {grid &&
        ySpec &&
        yTicks.map((t) => {
          const py = Math.round(yScale(t)) + 0.5;
          return (
            <line key={`grid-y-${t}`} x1={0} x2={innerWidth} y1={py} y2={py} stroke={tokens.color.gridLine} strokeWidth={1} />
          );
        })}

      {showXZero &&
        (() => {
          const px = Math.round(xScale(0)) + 0.5;
          return <line x1={px} x2={px} y1={0} y2={innerHeight} stroke={tokens.color.axisLine} strokeWidth={1} />;
        })()}
      {showYZero &&
        (() => {
          const py = Math.round(yScale(0)) + 0.5;
          return <line x1={0} x2={innerWidth} y1={py} y2={py} stroke={tokens.color.axisLine} strokeWidth={1} />;
        })()}

      {xSpec && !xSpec.bare && (
        <line x1={0} x2={innerWidth} y1={axisLineY} y2={axisLineY} stroke={tokens.color.axisLine} strokeWidth={1} />
      )}
      {ySpec && !ySpec.bare && <line x1={0.5} x2={0.5} y1={0} y2={innerHeight} stroke={tokens.color.axisLine} strokeWidth={1} />}

      {xSpec &&
        !xSpec.bare &&
        xTicks.map((t) => {
          const px = Math.round(xScale(t));
          return (
            <g key={`tick-x-${t}`}>
              <line x1={px + 0.5} x2={px + 0.5} y1={innerHeight} y2={innerHeight + TICK_LENGTH} stroke={tokens.color.axisLine} strokeWidth={1} />
              <text x={px} y={Math.round(innerHeight + TICK_LENGTH + LABEL_GAP)} textAnchor="middle" dominantBaseline="hanging" fill={tokens.color.inkMuted} style={FONT_STYLE}>
                {xFormat(t)}
              </text>
            </g>
          );
        })}

      {ySpec &&
        !ySpec.bare &&
        yTicks.map((t) => {
          const py = Math.round(yScale(t));
          return (
            <g key={`tick-y-${t}`}>
              <line x1={-TICK_LENGTH} x2={0} y1={py + 0.5} y2={py + 0.5} stroke={tokens.color.axisLine} strokeWidth={1} />
              <text x={Math.round(-TICK_LENGTH - LABEL_GAP)} y={py} textAnchor="end" dominantBaseline="middle" fill={tokens.color.inkMuted} style={FONT_STYLE}>
                {yFormat(t)}
              </text>
            </g>
          );
        })}

      {xSpec?.label && (
        <text
          x={Math.round(innerWidth / 2)}
          y={Math.round(innerHeight + TICK_LENGTH + LABEL_GAP + AXIS_LABEL_GAP)}
          textAnchor="middle"
          dominantBaseline="hanging"
          fill={tokens.color.inkMuted}
          style={FONT_STYLE}
        >
          {xSpec.label}
        </text>
      )}

      {ySpec?.label &&
        (() => {
          const lx = Math.round(-TICK_LENGTH - LABEL_GAP - AXIS_LABEL_GAP - 10);
          const ly = Math.round(innerHeight / 2);
          return (
            <text
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="middle"
              fill={tokens.color.inkMuted}
              transform={`rotate(-90, ${lx}, ${ly})`}
              style={FONT_STYLE}
            >
              {ySpec.label}
            </text>
          );
        })()}
    </g>
  );
}
