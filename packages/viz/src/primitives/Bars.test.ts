import { scaleLinear } from 'd3-scale';
import { describe, expect, it } from 'vitest';

import type { Frame } from '../frame.js';
import { barRect } from './Bars.js';

function testFrame(): Frame {
  const xScale = scaleLinear().domain([0, 4]).range([0, 400]);
  const yScale = scaleLinear().domain([-1, 1]).range([200, 0]);
  return {
    width: 400,
    height: 200,
    innerWidth: 400,
    innerHeight: 200,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    xScale,
    yScale,
    toPx: (x, y) => [xScale(x), yScale(y)] as const,
    toData: (px, py) => [xScale.invert(px), yScale.invert(py)] as const,
    dpr: 1,
    requestRedraw: () => {},
  };
}

describe('barRect', () => {
  const frame = testFrame();

  it('centres a vertical bar on its slot and spans from the baseline to the value', () => {
    const rect = barRect(frame, { at: 1, value: 0.5, color: '' }, 0.8, 0, 'vertical');

    expect(rect.x).toBeCloseTo(60, 10);
    expect(rect.width).toBeCloseTo(80, 10);
    expect(rect.y).toBeCloseTo(50, 10);
    expect(rect.height).toBeCloseTo(50, 10);
  });

  it('grows downwards from the baseline for a negative value rather than inverting', () => {
    const rect = barRect(frame, { at: 1, value: -0.5, color: '' }, 0.8, 0, 'vertical');

    expect(rect.y).toBeCloseTo(100, 10);
    expect(rect.height).toBeCloseTo(50, 10);
  });

  it('measures a vertical bar from a non-zero baseline', () => {
    const rect = barRect(frame, { at: 2, value: 1, color: '' }, 0.8, 0.5, 'vertical');

    expect(rect.y).toBeCloseTo(0, 10);
    expect(rect.height).toBeCloseTo(50, 10);
  });

  it('swaps the axes when horizontal', () => {
    const rect = barRect(frame, { at: 1, value: 2, color: '' }, 0.8, 0, 'horizontal');

    expect(rect.x).toBeCloseTo(0, 10);
    expect(rect.width).toBeCloseTo(200, 10);
    expect(rect.height).toBeCloseTo(80, 10);
  });
});
