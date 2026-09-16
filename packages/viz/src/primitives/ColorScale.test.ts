import { describe, expect, it } from 'vitest';

import { divergingScale, interpolateStops, oklabToSrgb, quantize, sequentialScale, srgbToOklab, withAlpha, cssColorToRgbaBytes } from './ColorScale.js';

function parseRgbaString(color: string): readonly [number, number, number, number] {
  const match = /^rgba\((\d+), (\d+), (\d+), ([0-9.]+)\)$/.exec(color);
  if (!match) throw new Error(`unexpected colour format: ${color}`);
  const [, r, g, b, a] = match;
  return [Number(r), Number(g), Number(b), Number(a)];
}

describe('srgbToOklab / oklabToSrgb round trip', () => {
  const samples: ReadonlyArray<readonly [number, number, number]> = [
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.5, 0.5, 0.5],
    [0.2, 0.7, 0.9],
    [0.9, 0.1, 0.4],
  ];

  it('reconstructs the original sRGB channels to within 1e-6', () => {
    const tolerance = 1e-6;
    for (const [r, g, b] of samples) {
      const [L, a, ok_b] = srgbToOklab(r, g, b);
      const [r2, g2, b2] = oklabToSrgb(L, a, ok_b);
      expect(Math.abs(r2 - r)).toBeLessThan(tolerance);
      expect(Math.abs(g2 - g)).toBeLessThan(tolerance);
      expect(Math.abs(b2 - b)).toBeLessThan(tolerance);
    }
  });
});

describe('interpolateStops', () => {
  it('returns the exact stop colours at the endpoints', () => {
    // #440154 = (68, 1, 84); #fde725 = (253, 231, 37).
    const interpolator = interpolateStops(['#440154', '#21908c', '#fde725']);
    expect(interpolator(0)).toBe('rgba(68, 1, 84, 1)');
    expect(interpolator(1)).toBe('rgba(253, 231, 37, 1)');
  });

  it('lands exactly on an interior stop when t divides evenly', () => {
    const interpolator = interpolateStops(['#000000', '#808080', '#ffffff']);
    expect(interpolator(0.5)).toBe('rgba(128, 128, 128, 1)');
  });

  it('clamps t outside [0, 1]', () => {
    const stops = ['#000000', '#ffffff'];
    const interpolator = interpolateStops(stops);
    expect(interpolator(-1)).toBe(interpolator(0));
    expect(interpolator(2)).toBe(interpolator(1));
  });
});

describe('sequentialScale', () => {
  it('maps domain endpoints to the interpolator endpoints', () => {
    const stops = ['#440154', '#fde725'];
    const scale = sequentialScale([0, 10], stops);
    expect(scale(0)).toBe(interpolateStops(stops)(0));
    expect(scale(10)).toBe(interpolateStops(stops)(1));
  });
});

describe('divergingScale', () => {
  it('is symmetric about its centre even when the domain is lopsided', () => {
    const stops = ['#ff0000', '#ffffff', '#ff0000'];
    const scale = divergingScale([-1, 5], 0, stops);
    expect(scale(-1)).toBe(scale(1));
    expect(scale(-0.5)).toBe(scale(0.5));
  });

  it('reaches the outer stop only at the side with more range', () => {
    const stops = ['#0000ff', '#ffffff', '#ff0000'];
    const scale = divergingScale([-1, 5], 0, stops);
    expect(scale(5)).toBe(interpolateStops(stops)(1));
    expect(scale(-1)).not.toBe(interpolateStops(stops)(0));
  });
});

describe('quantize', () => {
  const stops = ['#000000', '#ffffff'];
  const interpolator = interpolateStops(stops);

  it('n=1 returns a single representative midpoint sample', () => {
    const result = quantize(interpolator, 1);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(interpolator(0.5));
  });

  it('n=2 returns the two endpoint colours', () => {
    const result = quantize(interpolator, 2);
    expect(result).toEqual([interpolator(0), interpolator(1)]);
  });

  it('n=5 is evenly spaced including both endpoints', () => {
    const result = quantize(interpolator, 5);
    expect(result).toEqual([0, 0.25, 0.5, 0.75, 1].map((t) => interpolator(t)));
  });
});

describe('withAlpha', () => {
  it('parses #rgb', () => {
    expect(parseRgbaString(withAlpha('#0f0', 0.5))).toEqual([0, 255, 0, 0.5]);
  });

  it('parses #rrggbb', () => {
    expect(parseRgbaString(withAlpha('#0072B2', 0.4))).toEqual([0, 114, 178, 0.4]);
  });

  it('parses #rrggbbaa, ignoring the original alpha', () => {
    expect(parseRgbaString(withAlpha('#0072B280', 0.9))).toEqual([0, 114, 178, 0.9]);
  });

  it('parses rgb()', () => {
    expect(parseRgbaString(withAlpha('rgb(0, 114, 178)', 0.3))).toEqual([0, 114, 178, 0.3]);
  });

  it('parses rgba() with an existing alpha, which is overridden', () => {
    expect(parseRgbaString(withAlpha('rgba(0, 114, 178, 0.1)', 0.7))).toEqual([0, 114, 178, 0.7]);
  });

  it('parses modern space-separated rgb()', () => {
    expect(parseRgbaString(withAlpha('rgb(0 114 178)', 0.6))).toEqual([0, 114, 178, 0.6]);
  });

  it('parses oklch()', () => {
    const result = parseRgbaString(withAlpha('oklch(0.5 0.1 30)', 0.5));
    expect(result[3]).toBe(0.5);
    for (const channel of result.slice(0, 3)) {
      expect(channel).toBeGreaterThanOrEqual(0);
      expect(channel).toBeLessThanOrEqual(255);
    }
  });

  it('throws with the offending string when it cannot parse', () => {
    expect(() => withAlpha('not-a-colour', 0.5)).toThrow(/not-a-colour/);
    expect(() => withAlpha('hsl(0, 100%, 50%)', 0.5)).toThrow(/hsl\(0, 100%, 50%\)/);
  });
});

describe('cssColorToRgbaBytes', () => {
  it('parses 6-digit and 3-digit hex', () => {
    expect(cssColorToRgbaBytes('#000000')).toEqual([0, 0, 0, 255]);
    expect(cssColorToRgbaBytes('#ffffff')).toEqual([255, 255, 255, 255]);
    expect(cssColorToRgbaBytes('#f00')).toEqual([255, 0, 0, 255]);
  });

  it('parses 8-digit hex with alpha', () => {
    expect(cssColorToRgbaBytes('#ff000080')).toEqual([255, 0, 0, 128]);
  });

  it('parses rgb() and rgba()', () => {
    expect(cssColorToRgbaBytes('rgb(10, 20, 30)')).toEqual([10, 20, 30, 255]);
    expect(cssColorToRgbaBytes('rgba(10, 20, 30, 0.5)')).toEqual([10, 20, 30, 128]);
  });

  it('parses modern space-separated rgb with an alpha slash', () => {
    expect(cssColorToRgbaBytes('rgb(10 20 30 / 50%)')).toEqual([10, 20, 30, 128]);
  });

  it('parses oklch white and black at the extremes', () => {
    const [r, g, b, a] = cssColorToRgbaBytes('oklch(1 0 0)');
    expect(r).toBeGreaterThanOrEqual(254);
    expect(g).toBeGreaterThanOrEqual(254);
    expect(b).toBeGreaterThanOrEqual(254);
    expect(a).toBe(255);

    expect(cssColorToRgbaBytes('oklch(0 0 0)')).toEqual([0, 0, 0, 255]);
  });

  it('throws on an unrecognised format', () => {
    expect(() => cssColorToRgbaBytes('not-a-colour')).toThrow();
  });
});
