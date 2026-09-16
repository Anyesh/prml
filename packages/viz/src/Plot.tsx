import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { scaleLinear } from 'd3-scale';

import {
  DEFAULT_MARGIN,
  type CanvasDraw,
  type CanvasRegistry,
  type Domain,
  type Frame,
  type Margin,
} from './frame.js';

const FrameContext = createContext<Frame | null>(null);
const CanvasContext = createContext<CanvasRegistry | null>(null);

export function useFrame(): Frame {
  const frame = useContext(FrameContext);
  if (!frame) throw new Error('useFrame must be called inside a <Plot>');
  return frame;
}

/**
 * Enrols a canvas draw callback for the enclosing `<Plot>`. This is the only sanctioned
 * way to reach a 2D context in this codebase; a primitive that mounts its own `<canvas>`
 * gets its own stacking context and stops composing with the rest of the figure.
 *
 * `draw` is re-registered whenever `deps` change, and the figure repaints. The context is
 * already scaled by device pixel ratio and translated by the margin.
 */
export function useCanvasLayer(z: number, draw: CanvasDraw, deps: readonly unknown[]): void {
  const registry = useContext(CanvasContext);
  if (!registry) throw new Error('useCanvasLayer must be called inside a <Plot>');
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    const unregister = registry.register(z, (ctx, frame) => drawRef.current(ctx, frame));
    return unregister;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [registry, z, ...deps]);
}

interface Registration {
  readonly z: number;
  readonly seq: number;
  readonly draw: CanvasDraw;
}

export interface PlotProps {
  /** Omit to fill the container width, tracked with a ResizeObserver. */
  width?: number;
  height: number;
  xDomain: Domain;
  yDomain: Domain;
  margin?: Partial<Margin>;
  /**
   * Forces the same pixels-per-unit on both axes by widening whichever domain is
   * relatively narrower. Required for anything where shape carries meaning: covariance
   * ellipses, decision boundaries, PCA axes.
   */
  equalAspect?: boolean;
  /** Accessible description of the figure, announced in place of the canvas contents. */
  label?: string;
  /** Rendered inside the SVG overlay, so children must be SVG elements or return null. */
  children?: ReactNode;
}

export function Plot({
  width,
  height,
  xDomain,
  yDomain,
  margin: marginOverride,
  equalAspect = false,
  label,
  children,
}: PlotProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [measured, setMeasured] = useState(width ?? 0);
  const [dpr, setDpr] = useState(1);

  useLayoutEffect(() => {
    if (width !== undefined) {
      setMeasured(width);
      return;
    }
    const host = hostRef.current;
    if (!host) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setMeasured(Math.round(entry.contentRect.width));
    });
    observer.observe(host);
    setMeasured(Math.round(host.getBoundingClientRect().width));
    return () => observer.disconnect();
  }, [width]);

  useEffect(() => {
    const update = () => setDpr(Math.min(window.devicePixelRatio || 1, 2));
    update();
    const media = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  // Callers write these inline (`xDomain={[-1, 1]}`, `margin={{ left: 60 }}`), so a new
  // object arrives on every render. Depending on the identities would rebuild the frame
  // and repaint every canvas layer each time the parent renders, so the memos below must
  // depend on the scalar contents instead.
  const marginTop = marginOverride?.top;
  const marginRight = marginOverride?.right;
  const marginBottom = marginOverride?.bottom;
  const marginLeft = marginOverride?.left;
  const [x0Prop, x1Prop] = xDomain;
  const [y0Prop, y1Prop] = yDomain;

  const margin: Margin = useMemo(
    () => ({
      top: marginTop ?? DEFAULT_MARGIN.top,
      right: marginRight ?? DEFAULT_MARGIN.right,
      bottom: marginBottom ?? DEFAULT_MARGIN.bottom,
      left: marginLeft ?? DEFAULT_MARGIN.left,
    }),
    [marginTop, marginRight, marginBottom, marginLeft],
  );

  const registrations = useRef<Registration[]>([]);
  const seq = useRef(0);
  const rafId = useRef(0);
  const frameRef = useRef<Frame | null>(null);

  const paint = useCallback(() => {
    rafId.current = 0;
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame || frame.innerWidth <= 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(frame.dpr, frame.dpr);
    ctx.translate(frame.margin.left, frame.margin.top);
    const ordered = [...registrations.current].sort((a, b) => a.z - b.z || a.seq - b.seq);
    for (const layer of ordered) {
      ctx.save();
      layer.draw(ctx, frame);
      ctx.restore();
    }
    ctx.restore();
  }, []);

  const requestRedraw = useCallback(() => {
    if (rafId.current) return;
    rafId.current = requestAnimationFrame(paint);
  }, [paint]);

  const registry = useMemo<CanvasRegistry>(
    () => ({
      register(z, draw) {
        const entry: Registration = { z, seq: seq.current++, draw };
        registrations.current = [...registrations.current, entry];
        requestRedraw();
        return () => {
          registrations.current = registrations.current.filter((r) => r !== entry);
          requestRedraw();
        };
      },
    }),
    [requestRedraw],
  );

  const frame = useMemo<Frame>(() => {
    const outerWidth = measured;
    const innerWidth = Math.max(0, outerWidth - margin.left - margin.right);
    const innerHeight = Math.max(0, height - margin.top - margin.bottom);

    let x0 = x0Prop;
    let x1 = x1Prop;
    let y0 = y0Prop;
    let y1 = y1Prop;
    if (equalAspect && innerWidth > 0 && innerHeight > 0) {
      const xPerPx = (x1 - x0) / innerWidth;
      const yPerPx = (y1 - y0) / innerHeight;
      const unit = Math.max(xPerPx, yPerPx);
      const xPad = (unit * innerWidth - (x1 - x0)) / 2;
      const yPad = (unit * innerHeight - (y1 - y0)) / 2;
      x0 -= xPad;
      x1 += xPad;
      y0 -= yPad;
      y1 += yPad;
    }

    const xScale = scaleLinear().domain([x0, x1]).range([0, innerWidth]);
    const yScale = scaleLinear().domain([y0, y1]).range([innerHeight, 0]);

    return {
      width: outerWidth,
      height,
      innerWidth,
      innerHeight,
      margin,
      xScale,
      yScale,
      dpr,
      toPx: (x, y) => [xScale(x), yScale(y)] as const,
      toData: (px, py) => [xScale.invert(px), yScale.invert(py)] as const,
      requestRedraw,
    };
  }, [measured, height, x0Prop, x1Prop, y0Prop, y1Prop, margin, equalAspect, dpr, requestRedraw]);

  frameRef.current = frame;

  useLayoutEffect(() => {
    requestRedraw();
  }, [frame, requestRedraw]);

  useEffect(() => () => cancelAnimationFrame(rafId.current), []);

  return (
    <div ref={hostRef} className="prml-plot" style={{ position: 'relative', height }}>
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        width={Math.round(frame.width * dpr)}
        height={Math.round(height * dpr)}
        style={{ position: 'absolute', inset: 0, width: frame.width, height }}
      />
      <svg
        role="img"
        aria-label={label ?? 'figure'}
        width={frame.width}
        height={height}
        style={{ position: 'absolute', inset: 0, overflow: 'visible' }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {frame.innerWidth > 0 ? (
            <FrameContext.Provider value={frame}>
              <CanvasContext.Provider value={registry}>{children}</CanvasContext.Provider>
            </FrameContext.Provider>
          ) : null}
        </g>
      </svg>
    </div>
  );
}
