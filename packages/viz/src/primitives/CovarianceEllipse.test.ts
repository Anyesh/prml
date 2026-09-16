import { scaleLinear } from 'd3-scale';
import { inverse } from '@prml/math';
import type { Mat, Vec } from '@prml/math';
import { describe, expect, it } from 'vitest';

import type { Frame } from '../frame.js';
import { assertEqualAspect, covarianceEllipsePoints, ELLIPSE_SEGMENTS } from './CovarianceEllipse.js';

function mahalanobis(point: readonly [number, number], mean: Vec, cov: Mat): number {
  const inv = inverse(cov);
  const d = [point[0] - mean[0]!, point[1] - mean[1]!];
  let quad = 0;
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) quad += d[i]! * inv[i]![j]! * d[j]!;
  }
  return quad;
}

function testFrame(xDomain: [number, number], yDomain: [number, number], innerWidth: number, innerHeight: number): Frame {
  const xScale = scaleLinear().domain(xDomain).range([0, innerWidth]);
  const yScale = scaleLinear().domain(yDomain).range([innerHeight, 0]);
  return {
    width: innerWidth,
    height: innerHeight,
    innerWidth,
    innerHeight,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    xScale,
    yScale,
    toPx: (x, y) => [xScale(x), yScale(y)] as const,
    toData: (px, py) => [xScale.invert(px), yScale.invert(py)] as const,
    dpr: 1,
    requestRedraw: () => {},
  };
}

describe('covarianceEllipsePoints', () => {
  const mean = [-1.5, 2.25];
  const cov = [
    [2, 1],
    [1, 1.5],
  ];

  it('places every sample at the Mahalanobis radius the enclosed mass implies', () => {
    const mass = 0.9;
    const expected = -2 * Math.log(1 - mass);

    for (const point of covarianceEllipsePoints(mean, cov, mass)) {
      expect(mahalanobis(point, mean, cov)).toBeCloseTo(expected, 9);
    }
  });

  it('separates nested levels by their radii rather than redrawing one contour', () => {
    const inner = covarianceEllipsePoints(mean, cov, 0.5);
    const outer = covarianceEllipsePoints(mean, cov, 0.99);

    expect(mahalanobis(outer[0]!, mean, cov)).toBeGreaterThan(mahalanobis(inner[0]!, mean, cov));
    expect(mahalanobis(outer[0]!, mean, cov)).toBeCloseTo(-2 * Math.log(0.01), 9);
  });

  it('closes the path so the stroke has no seam', () => {
    const points = covarianceEllipsePoints(mean, cov, 0.9);

    expect(points).toHaveLength(ELLIPSE_SEGMENTS + 1);
    expect(points[points.length - 1]![0]).toBeCloseTo(points[0]![0]!, 12);
    expect(points[points.length - 1]![1]).toBeCloseTo(points[0]![1]!, 12);
  });

  it('orients the first sample along the major axis, not along x', () => {
    const points = covarianceEllipsePoints(mean, cov, 0.9);
    const offset = [points[0]![0] - mean[0]!, points[0]![1] - mean[1]!];
    const mapped = [
      cov[0]![0]! * offset[0]! + cov[0]![1]! * offset[1]!,
      cov[1]![0]! * offset[0]! + cov[1]![1]! * offset[1]!,
    ];

    expect(mapped[0]! * offset[1]! - mapped[1]! * offset[0]!).toBeCloseTo(0, 9);
    expect(offset[1]).not.toBeCloseTo(0, 3);

    const longest = Math.max(...points.map(([x, y]) => Math.hypot(x - mean[0]!, y - mean[1]!)));
    expect(Math.hypot(offset[0]!, offset[1]!)).toBeCloseTo(longest, 9);
  });

  it('draws a circle for an isotropic covariance', () => {
    const points = covarianceEllipsePoints([0, 0], [[4, 0], [0, 4]], 1 - Math.exp(-0.5));

    for (const [x, y] of points) {
      expect(Math.hypot(x, y)).toBeCloseTo(2, 9);
    }
  });
});

describe('assertEqualAspect', () => {
  it('accepts a frame whose axes share pixels-per-unit', () => {
    expect(() => assertEqualAspect(testFrame([-4, 4], [-2, 2], 400, 200), 'CovarianceEllipse')).not.toThrow();
  });

  it('rejects a frame that would stretch the shape, naming the prop that fixes it', () => {
    expect(() => assertEqualAspect(testFrame([-4, 4], [-4, 4], 400, 200), 'CovarianceEllipse')).toThrow(/equalAspect/);
  });

  it('tolerates the sub-percent slack a rounded pixel width leaves behind', () => {
    expect(() => assertEqualAspect(testFrame([-4, 4], [-2, 2.004], 400, 200), 'CovarianceEllipse')).not.toThrow();
  });
});
