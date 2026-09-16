import { describe, expect, it } from 'vitest';

import { boundaryOffset, edgeGeometry, layeredLayout } from './NetworkDiagram.js';

describe('boundaryOffset', () => {
  it('is the radius in every direction for a circle', () => {
    const radius = 12;
    const directions: (readonly [number, number])[] = [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
      [Math.SQRT1_2, Math.SQRT1_2],
      [-Math.SQRT1_2, Math.SQRT1_2],
    ];

    for (const [ux, uy] of directions) {
      expect(boundaryOffset('circle', radius, ux, uy)).toBeCloseTo(radius, 10);
    }
  });

  it('is the half-side along an axis for a square', () => {
    const half = 10;
    expect(boundaryOffset('square', half, 1, 0)).toBeCloseTo(half, 10);
    expect(boundaryOffset('square', half, 0, 1)).toBeCloseTo(half, 10);
    expect(boundaryOffset('square', half, -1, 0)).toBeCloseTo(half, 10);
  });

  it('is larger on the diagonal than along an axis for a square', () => {
    const half = 10;
    const axis = boundaryOffset('square', half, 1, 0);
    const diagonal = boundaryOffset('square', half, Math.SQRT1_2, Math.SQRT1_2);
    expect(diagonal).toBeGreaterThan(axis);
    expect(diagonal).toBeCloseTo(half * Math.SQRT2, 10);
  });
});

describe('edgeGeometry', () => {
  it('trims to exactly the radius from each centre and stays collinear', () => {
    const from = { px: 0, py: 0, shape: 'circle' as const, radius: 5 };
    const to = { px: 30, py: 40, shape: 'circle' as const, radius: 8 };

    const geometry = edgeGeometry(from, to, false);

    const distFrom = Math.hypot(geometry.x1 - from.px, geometry.y1 - from.py);
    const distTo = Math.hypot(geometry.x2 - to.px, geometry.y2 - to.py);
    expect(distFrom).toBeCloseTo(5, 10);
    expect(distTo).toBeCloseTo(8, 10);

    const crossStart =
      (geometry.x1 - from.px) * (to.py - from.py) - (geometry.y1 - from.py) * (to.px - from.px);
    const crossEnd =
      (geometry.x2 - from.px) * (to.py - from.py) - (geometry.y2 - from.py) * (to.px - from.px);
    expect(crossStart).toBeCloseTo(0, 10);
    expect(crossEnd).toBeCloseTo(0, 10);
  });

  it('has a null arrow when undirected', () => {
    const from = { px: 0, py: 0, shape: 'circle' as const, radius: 5 };
    const to = { px: 10, py: 0, shape: 'circle' as const, radius: 5 };

    const geometry = edgeGeometry(from, to, false);
    expect(geometry.arrow).toBeNull();
  });

  it('has a three-point arrow triangle when directed', () => {
    const from = { px: 0, py: 0, shape: 'circle' as const, radius: 5 };
    const to = { px: 100, py: 0, shape: 'circle' as const, radius: 5 };

    const geometry = edgeGeometry(from, to, true);
    expect(geometry.arrow).not.toBeNull();
    expect(geometry.arrow).toHaveLength(3);
    expect(geometry.arrow![0]![0]).toBeCloseTo(geometry.x2, 10);
    expect(geometry.arrow![0]![1]).toBeCloseTo(geometry.y2, 10);
  });
});

describe('layeredLayout', () => {
  it('centres a single-unit layer on the vertical midline', () => {
    const nodes = layeredLayout([1, 1]);
    const mid = 0.5;
    for (const node of nodes) {
      expect(node.y).toBeCloseTo(mid, 10);
    }
  });

  it('spaces a three-unit layer evenly across the y box', () => {
    const nodes = layeredLayout([3], { x0: 0, x1: 1, y0: 0, y1: 1 });
    const ys = nodes.map((n) => n.y).sort((a, b) => a - b);
    expect(ys[0]).toBeCloseTo(0, 10);
    expect(ys[1]).toBeCloseTo(0.5, 10);
    expect(ys[2]).toBeCloseTo(1, 10);
  });

  it('spreads layers evenly across the x box and ids them L<layer>N<index>', () => {
    const nodes = layeredLayout([2, 2]);
    const layer0 = nodes.filter((n) => n.layer === 0);
    const layer1 = nodes.filter((n) => n.layer === 1);
    expect(layer0.every((n) => n.x === 0)).toBe(true);
    expect(layer1.every((n) => n.x === 1)).toBe(true);
    expect(nodes.map((n) => n.id)).toEqual(['L0N0', 'L0N1', 'L1N0', 'L1N1']);
  });
});
