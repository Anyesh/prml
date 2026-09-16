import { describe, expect, it, vi } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { StepThrough, clampStep, isAtEnd, isAtStart } from './StepThrough.js';

describe('clampStep', () => {
  it('bounds into [0, stepCount - 1]', () => {
    expect(clampStep(-2, 5)).toBe(0);
    expect(clampStep(2, 5)).toBe(2);
    expect(clampStep(9, 5)).toBe(4);
  });

  it('bottoms out at 0 for an empty sequence', () => {
    expect(clampStep(3, 0)).toBe(0);
  });
});

describe('isAtStart / isAtEnd', () => {
  it('flags the first and last valid index', () => {
    expect(isAtStart(0)).toBe(true);
    expect(isAtStart(1)).toBe(false);
    expect(isAtEnd(4, 5)).toBe(true);
    expect(isAtEnd(3, 5)).toBe(false);
  });
});

describe('StepThrough markup', () => {
  it('hides the play button when interval is omitted', () => {
    const html = renderToStaticMarkup(
      createElement(StepThrough, { step: 0, stepCount: 4, onStep: vi.fn() }),
    );
    expect(html).not.toContain('aria-label="Play"');
    expect(html).toContain('aria-label="Previous step"');
    expect(html).toContain('aria-label="Next step"');
    expect(html).toContain('aria-label="Reset"');
  });

  it('shows the play button when interval is given', () => {
    const html = renderToStaticMarkup(
      createElement(StepThrough, { step: 0, stepCount: 4, onStep: vi.fn(), interval: 800 }),
    );
    expect(html).toContain('aria-label="Play"');
  });

  it('disables back at the first step and forward at the last step', () => {
    const first = renderToStaticMarkup(
      createElement(StepThrough, { step: 0, stepCount: 3, onStep: vi.fn() }),
    );
    expect(first).toMatch(/disabled="[^>]*aria-label="Previous step"/);

    const last = renderToStaticMarkup(
      createElement(StepThrough, { step: 2, stepCount: 3, onStep: vi.fn() }),
    );
    expect(last).toMatch(/disabled="[^>]*aria-label="Next step"/);
  });

  it('shows the current step label when provided', () => {
    const html = renderToStaticMarkup(
      createElement(StepThrough, {
        step: 1,
        stepCount: 2,
        onStep: vi.fn(),
        labels: ['E-step', 'M-step'],
      }),
    );
    expect(html).toContain('M-step');
    expect(html).toContain('Step 2 of 2');
  });
});
