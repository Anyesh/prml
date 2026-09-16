/**
 * The part of a d3 scale the primitives actually use. Narrower than `ScaleLinear` on purpose:
 * `scaleLog` satisfies this too, so a log axis needs no change anywhere downstream of `Plot`.
 */
export interface PlotScale {
  (value: number): number;
  invert(px: number): number;
  domain(): number[];
  ticks(count?: number): number[];
  tickFormat(count?: number, specifier?: string): (value: number | { valueOf(): number }) => string;
}

export type ScaleKind = 'linear' | 'log';

export interface Margin {
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
  readonly left: number;
}

export type Domain = readonly [number, number];

/**
 * The coordinate system every primitive inside a `<Plot>` shares, published through React
 * context. Primitives never compute their own scales: two primitives with independently
 * derived scales drift by a pixel and the figure stops lining up.
 *
 * All pixel coordinates are *inner* coordinates, with the origin at the top-left of the
 * plotting area rather than of the element. Both the SVG group and the canvas transform
 * are pre-translated by the margin, so a primitive must not add it again.
 */
export interface Frame {
  readonly width: number;
  readonly height: number;
  readonly innerWidth: number;
  readonly innerHeight: number;
  readonly margin: Margin;
  readonly xScale: PlotScale;
  readonly yScale: PlotScale;
  /** Data coordinates to inner pixels. */
  readonly toPx: (x: number, y: number) => readonly [number, number];
  /** Inner pixels back to data coordinates, for pointer handling. */
  readonly toData: (px: number, py: number) => readonly [number, number];
  /**
   * Device pixel ratio the canvas backing store is scaled by. Drawing code works in CSS
   * pixels and must not multiply by this, but a primitive that wants a hairline can set
   * `lineWidth = 1 / dpr` to get one physical pixel.
   */
  readonly dpr: number;
  /**
   * Schedules a canvas repaint on the next animation frame. Coalesces, so calling it
   * from ten primitives in one React commit repaints once.
   */
  readonly requestRedraw: () => void;
}

export type CanvasDraw = (ctx: CanvasRenderingContext2D, frame: Frame) => void;

/**
 * Registry the `<Plot>` root exposes so canvas primitives can enrol a draw callback.
 * Kept separate from `Frame` because `Frame` changes on every resize while this does not,
 * and merging them would repaint the whole figure on each React render.
 */
export interface CanvasRegistry {
  /**
   * Layers must paint in ascending `z`, with ties broken by registration order, because
   * a contour field drawn after the curve it sits behind hides it. The returned function
   * unregisters.
   */
  readonly register: (z: number, draw: CanvasDraw) => () => void;
}

export const DEFAULT_MARGIN: Margin = { top: 12, right: 16, bottom: 34, left: 46 };

export class NotImplemented extends Error {
  constructor(what: string) {
    super(`@prml/viz: ${what} is not implemented yet`);
    this.name = 'NotImplemented';
  }
}
