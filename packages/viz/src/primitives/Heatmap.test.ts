import { describe, expect, it } from 'vitest';

import { buildColorLut, lutIndexForValue } from './Heatmap.js';

describe('lutIndexForValue', () => {
  it('maps the domain endpoints to the first and last bucket', () => {
    expect(lutIndexForValue(0, [0, 10], 11)).toBe(0);
    expect(lutIndexForValue(10, [0, 10], 11)).toBe(10);
    expect(lutIndexForValue(5, [0, 10], 11)).toBe(5);
  });

  it('clamps values outside the domain', () => {
    expect(lutIndexForValue(-5, [0, 10], 11)).toBe(0);
    expect(lutIndexForValue(15, [0, 10], 11)).toBe(10);
  });

  it('returns bucket 0 for a degenerate domain or a non-finite value', () => {
    expect(lutIndexForValue(5, [3, 3], 11)).toBe(0);
    expect(lutIndexForValue(NaN, [0, 10], 11)).toBe(0);
  });
});

describe('buildColorLut', () => {
  it('samples the interpolator once per bucket and stores parsed RGBA bytes in order', () => {
    const calls: number[] = [];
    const interpolator = (t: number) => {
      calls.push(t);
      return t < 5 ? '#000000' : '#ffffff';
    };

    const lut = buildColorLut(interpolator, [0, 10], 4);

    expect(calls).toHaveLength(4);
    [0, 10 / 3, 20 / 3, 10].forEach((expected, i) => expect(calls[i]).toBeCloseTo(expected, 10));
    expect(lut.length).toBe(16);
    expect(Array.from(lut.slice(0, 4))).toEqual([0, 0, 0, 255]);
    expect(Array.from(lut.slice(4, 8))).toEqual([0, 0, 0, 255]);
    expect(Array.from(lut.slice(8, 12))).toEqual([255, 255, 255, 255]);
    expect(Array.from(lut.slice(12, 16))).toEqual([255, 255, 255, 255]);
  });

  it('samples a single-bucket table at the domain minimum', () => {
    const interpolator = (t: number) => (t === 0 ? '#123456' : '#000000');
    const lut = buildColorLut(interpolator, [0, 10], 1);
    expect(Array.from(lut)).toEqual([0x12, 0x34, 0x56, 255]);
  });
});
