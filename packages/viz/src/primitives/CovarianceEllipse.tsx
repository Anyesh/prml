import { mvnCovarianceEllipse } from '@prml/math';
import type { Mat, Vec } from '@prml/math';

import type { Frame } from '../frame.js';
import { useFrame } from '../Plot.js';
import { Curve, type Dash } from './Curve.js';

/**
 * One sampling resolution for every contour in the book, so the same Gaussian does not
 * arrive slightly rounder in one chapter than another. At 72 the chord error on a 150px
 * radius is under a sixth of a pixel, well below what a reader can see.
 */
export const ELLIPSE_SEGMENTS = 72;

/** Relative slack between the two axes' pixels-per-unit before the shape counts as stretched. */
const ASPECT_TOLERANCE = 0.01;

/**
 * Samples the iso-density contour of a 2-D Gaussian enclosing `mass` of the probability
 * into a closed polyline. The radius and the axis angle come from `mvnCovarianceEllipse`
 * rather than being derived here, because the chi-squared quantile behind them is a
 * property of the distribution and must not have a second definition in the viz layer.
 */
export function covarianceEllipsePoints(
  mean: Vec,
  cov: Mat,
  mass: number,
  segments: number = ELLIPSE_SEGMENTS,
): (readonly [number, number])[] {
  const { cx, cy, rx, ry, angle } = mvnCovarianceEllipse({ mean, cov }, mass);
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  return Array.from({ length: segments + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / segments;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * cosA - y * sinA, cy + x * sinA + y * cosA] as const;
  });
}

/**
 * Throws unless both axes carry the same pixels-per-unit, which `<Plot equalAspect>` is
 * what guarantees. An ellipse on stretched axes is a different ellipse: its angle no
 * longer reads as the correlation and its axis ratio no longer reads as the eigenvalue
 * ratio, and because the result is still a plausible-looking ellipse the mistake survives
 * review. Any primitive whose shape carries meaning can reuse this.
 */
export function assertEqualAspect(frame: Frame, primitive: string): void {
  const xDomain = frame.xScale.domain();
  const yDomain = frame.yScale.domain();
  const xPerPx = Math.abs(xDomain[1]! - xDomain[0]!) / frame.innerWidth;
  const yPerPx = Math.abs(yDomain[1]! - yDomain[0]!) / frame.innerHeight;
  if (Math.abs(xPerPx - yPerPx) > ASPECT_TOLERANCE * Math.max(xPerPx, yPerPx)) {
    throw new Error(
      `@prml/viz: <${primitive}> needs a <Plot equalAspect>, but this plot shows ${xPerPx.toPrecision(3)} ` +
        `x-units and ${yPerPx.toPrecision(3)} y-units per pixel, which would stretch the shape.`,
    );
  }
}

export interface CovarianceEllipseProps {
  mean: Vec;
  /** 2x2 and positive definite. Not semi-axes: the eigendecomposition happens here. */
  cov: Mat;
  /** Probability masses, one contour drawn per entry. */
  levels?: readonly number[];
  /** A resolved colour from `useResolvedTokens`, never a literal. */
  color: string;
  width?: number;
  dash?: Dash;
  opacity?: number;
  z?: number;
}

/**
 * The confidence contours of a 2-D Gaussian, drawn from its covariance. Named for what it
 * draws rather than `Ellipse`, which `@prml/math` already uses for the geometry this is
 * built on and which several consuming widgets import under that name.
 */
export function CovarianceEllipse({
  mean,
  cov,
  levels = [0.95],
  color,
  width = 1.5,
  dash,
  opacity = 1,
  z = 0,
}: CovarianceEllipseProps) {
  assertEqualAspect(useFrame(), 'CovarianceEllipse');

  return (
    <>
      {levels.map((mass) => (
        <Curve
          key={mass}
          points={covarianceEllipsePoints(mean, cov, mass)}
          color={color}
          width={width}
          {...(dash ? { dash } : {})}
          opacity={opacity}
          z={z}
        />
      ))}
    </>
  );
}
