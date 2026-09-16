import type { Ellipse } from '@prml/math';

/**
 * `Curve` takes a polyline, not an ellipse primitive, so every Gaussian-contour figure in
 * this chapter converts through this one function rather than each re-deriving the
 * parametric rotation.
 */
export function ellipseToPolyline(ellipse: Ellipse, segments = 72): (readonly [number, number])[] {
  const { cx, cy, rx, ry, angle } = ellipse;
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  return Array.from({ length: segments + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / segments;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * cosA - y * sinA, cy + x * sinA + y * cosA] as const;
  });
}
