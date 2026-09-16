import { useFrame, usePlotClick } from '../index.js';

export interface ClickSurfaceProps {
  onClick: (x: number, y: number) => void;
  cursor?: string;
}

/**
 * A transparent hit target covering the plotting area. It must be the first SVG child of
 * a `<Plot>`, because anything rendered before it sits underneath and stops receiving
 * pointer events.
 */
export function ClickSurface({ onClick, cursor = 'crosshair' }: ClickSurfaceProps) {
  const frame = useFrame();
  const handle = usePlotClick(onClick);
  return (
    <rect
      x={0}
      y={0}
      width={frame.innerWidth}
      height={frame.innerHeight}
      fill="transparent"
      style={{ cursor, touchAction: 'none' }}
      onPointerDown={handle}
    />
  );
}
