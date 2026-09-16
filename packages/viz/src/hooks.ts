import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { DIVERGING, LIGHT, SEQUENTIAL, SERIES, colorVar, type ColorScheme } from '@prml/ui';

import { useFrame } from './Plot.js';

export interface ResolvedTokens {
  readonly color: Readonly<Record<keyof ColorScheme, string>>;
  readonly series: readonly string[];
  readonly sequential: readonly string[];
  readonly diverging: readonly string[];
}

const ROLES = Object.keys(LIGHT) as (keyof ColorScheme)[];

const FALLBACK: ResolvedTokens = {
  color: LIGHT,
  series: SERIES,
  sequential: SEQUENTIAL,
  diverging: DIVERGING,
};

function readTokens(): ResolvedTokens {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  const color = {} as Record<keyof ColorScheme, string>;
  for (const role of ROLES) color[role] = read(colorVar(role), LIGHT[role]);
  return {
    color,
    series: SERIES.map((f, i) => read(`--prml-series-${i}`, f)),
    sequential: SEQUENTIAL.map((f, i) => read(`--prml-sequential-${i}`, f)),
    diverging: DIVERGING.map((f, i) => read(`--prml-diverging-${i}`, f)),
  };
}

/**
 * Resolves the design tokens to literal colour strings for canvas drawing, and re-resolves
 * when the theme changes. Canvas has no access to CSS custom properties, so without this a
 * figure would keep its light-mode palette after a switch to dark, which is the single
 * most visible way a hand-rolled canvas betrays itself.
 */
export function useResolvedTokens(): ResolvedTokens {
  const [tokens, setTokens] = useState<ResolvedTokens>(FALLBACK);
  const frame = useFrame();

  useEffect(() => {
    const refresh = () => {
      setTokens(readTokens());
      frame.requestRedraw();
    };
    refresh();

    const observer = new MutationObserver(refresh);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener('change', refresh);
    return () => {
      observer.disconnect();
      media.removeEventListener('change', refresh);
    };
  }, [frame]);

  return tokens;
}

export interface DragOptions {
  onStart?: (x: number, y: number) => void;
  onDrag: (x: number, y: number) => void;
  onEnd?: (x: number, y: number) => void;
}

export interface DragBindings {
  readonly onPointerDown: (event: ReactPointerEvent<SVGElement>) => void;
  readonly style: { cursor: string; touchAction: 'none' };
}

/**
 * Pointer drag on an SVG child of `<Plot>`, reported in data coordinates.
 *
 * Capture is taken on the element so that a fast drag which outruns the pointer keeps
 * tracking, and `touch-action: none` is part of the returned style because without it
 * a touch drag scrolls the page instead of moving the point.
 */
export function useDrag(options: DragOptions): DragBindings {
  const frame = useFrame();
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<SVGElement>) => {
      event.preventDefault();
      event.stopPropagation();
      const target = event.currentTarget;
      const owner = target.ownerSVGElement ?? (target as unknown as SVGSVGElement);
      target.setPointerCapture(event.pointerId);

      const toData = (clientX: number, clientY: number) => {
        const rect = owner.getBoundingClientRect();
        return frame.toData(
          clientX - rect.left - frame.margin.left,
          clientY - rect.top - frame.margin.top,
        );
      };

      const start = toData(event.clientX, event.clientY);
      optionsRef.current.onStart?.(start[0], start[1]);

      const move = (e: PointerEvent) => {
        const [x, y] = toData(e.clientX, e.clientY);
        optionsRef.current.onDrag(x, y);
      };
      const up = (e: PointerEvent) => {
        const [x, y] = toData(e.clientX, e.clientY);
        optionsRef.current.onEnd?.(x, y);
        target.releasePointerCapture(e.pointerId);
        target.removeEventListener('pointermove', move);
        target.removeEventListener('pointerup', up);
        target.removeEventListener('pointercancel', up);
      };

      target.addEventListener('pointermove', move);
      target.addEventListener('pointerup', up);
      target.addEventListener('pointercancel', up);
    },
    [frame],
  );

  return { onPointerDown, style: { cursor: 'grab', touchAction: 'none' } };
}

/**
 * A transparent rectangle covering the plotting area that reports clicks in data
 * coordinates. Rendered by `<ClickSurface>`; this hook supplies its handler so widgets
 * that add a data point per click do not each re-derive the pixel arithmetic.
 */
export function usePlotClick(onClick: (x: number, y: number) => void) {
  const frame = useFrame();
  const handler = useRef(onClick);
  handler.current = onClick;

  return useCallback(
    (event: ReactPointerEvent<SVGRectElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      const [x, y] = frame.toData(event.clientX - rect.left, event.clientY - rect.top);
      handler.current(x, y);
    },
    [frame],
  );
}
