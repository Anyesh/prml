import { describe, expect, it, vi } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  Slider,
  clamp,
  defaultFormat,
  fractionToValue,
  nextValue,
  quantize,
  valueToFraction,
} from './Slider.js';

describe('clamp', () => {
  it('bounds a value into [min, max]', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-5, 0, 10)).toBe(0);
    expect(clamp(15, 0, 10)).toBe(10);
  });
});

describe('defaultFormat', () => {
  it('renders integers without a decimal point', () => {
    expect(defaultFormat(4)).toBe('4');
  });

  it('rounds to three decimals', () => {
    expect(defaultFormat(1 / 3)).toBe('0.333');
  });
});

describe('quantize', () => {
  it('snaps onto the nearest step from min', () => {
    expect(quantize(0.27, 0, 0.1)).toBeCloseTo(0.3);
    expect(quantize(2.6, 2, 0.5)).toBeCloseTo(2.5);
  });

  it('is a no-op for a non-positive step', () => {
    expect(quantize(1.234, 0, 0)).toBe(1.234);
  });
});

describe('valueToFraction / fractionToValue (linear)', () => {
  it('round-trips through the middle of the range', () => {
    expect(valueToFraction(5, 0, 10, 'linear')).toBeCloseTo(0.5);
    expect(fractionToValue(0.5, 0, 10, 'linear')).toBeCloseTo(5);
  });

  it('clamps out-of-range values before mapping', () => {
    expect(valueToFraction(-5, 0, 10, 'linear')).toBe(0);
    expect(valueToFraction(50, 0, 10, 'linear')).toBe(1);
  });
});

describe('valueToFraction / fractionToValue (log)', () => {
  it('maps the geometric midpoint to fraction 0.5', () => {
    const mid = Math.sqrt(1 * 100);
    expect(valueToFraction(mid, 1, 100, 'log')).toBeCloseTo(0.5);
    expect(fractionToValue(0.5, 1, 100, 'log')).toBeCloseTo(mid);
  });

  it('keeps min/max in real units at the travel extremes', () => {
    expect(fractionToValue(0, 1e-4, 10, 'log')).toBeCloseTo(1e-4);
    expect(fractionToValue(1, 1e-4, 10, 'log')).toBeCloseTo(10);
  });

  it('rejects a non-positive minimum', () => {
    expect(() => valueToFraction(1, 0, 10, 'log')).toThrow(/log\(0\)/);
    expect(() => fractionToValue(0.5, -1, 10, 'log')).toThrow(/log\(0\)/);
  });
});

describe('nextValue', () => {
  it('adds an explicit step in real units and quantizes to it', () => {
    expect(nextValue(1, 1, 0, 10, 'linear', 0.5, false)).toBeCloseTo(1.5);
    expect(nextValue(1.2, 1, 0, 10, 'linear', 0.5, false)).toBeCloseTo(1.5);
  });

  it('multiplies the explicit step by ten under a coarse (shift) step', () => {
    expect(nextValue(1, 1, 0, 100, 'linear', 1, true)).toBeCloseTo(11);
  });

  it('clamps at the range boundary', () => {
    expect(nextValue(9.6, 1, 0, 10, 'linear', 1, false)).toBe(10);
    expect(nextValue(0.4, -1, 0, 10, 'linear', 1, false)).toBe(0);
  });

  it('divides the travel into equal linear steps when no step is given', () => {
    const step = nextValue(0, 1, 0, 100, 'linear', undefined, false) - 0;
    expect(step).toBeCloseTo(1);
  });

  it('moves by a constant ratio (not a constant amount) on a log scale with no step', () => {
    const from10 = nextValue(10, 1, 1, 1000, 'log', undefined, false) / 10;
    const from100 = nextValue(100, 1, 1, 1000, 'log', undefined, false) / 100;
    expect(from10).toBeCloseTo(from100, 6);
  });
});

describe('Slider markup', () => {
  it('exposes the ARIA slider contract with a formatted value', () => {
    const html = renderToStaticMarkup(
      createElement(Slider, {
        label: 'Learning rate',
        value: 0.1,
        min: 0,
        max: 1,
        onChange: vi.fn(),
        hint: 'Higher values converge faster but can overshoot.',
      }),
    );
    expect(html).toContain('role="slider"');
    expect(html).toContain('aria-valuemin="0"');
    expect(html).toContain('aria-valuemax="1"');
    expect(html).toContain('aria-valuenow="0.1"');
    expect(html).toContain('Learning rate');
    expect(html).toContain('Higher values converge faster but can overshoot.');
  });

  it('marks a disabled slider as unfocusable and aria-disabled', () => {
    const html = renderToStaticMarkup(
      createElement(Slider, {
        label: 'Alpha',
        value: 1,
        min: 0,
        max: 10,
        onChange: vi.fn(),
        disabled: true,
      }),
    );
    expect(html).toContain('tabindex="-1"');
    expect(html).toContain('aria-disabled="true"');
  });

  it('uses a caller-supplied format for both the readout and aria-valuetext', () => {
    const html = renderToStaticMarkup(
      createElement(Slider, {
        label: 'Sigma',
        value: 2,
        min: 0,
        max: 10,
        onChange: vi.fn(),
        format: (v) => `${v.toFixed(2)}σ`,
      }),
    );
    expect(html).toContain('2.00σ');
  });
});
