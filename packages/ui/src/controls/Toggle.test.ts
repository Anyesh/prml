import { describe, expect, it, vi } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Toggle } from './Toggle.js';

describe('Toggle markup', () => {
  it('renders a real checkbox exposed as a switch', () => {
    const html = renderToStaticMarkup(
      createElement(Toggle, { label: 'Show noise', checked: false, onChange: vi.fn() }),
    );
    expect(html).toContain('type="checkbox"');
    expect(html).toContain('role="switch"');
    expect(html).toContain('Show noise');
  });

  it('reflects checked state on the input', () => {
    const html = renderToStaticMarkup(
      createElement(Toggle, { label: 'Show noise', checked: true, onChange: vi.fn() }),
    );
    expect(html).toContain('checked=""');
  });

  it('wires the hint to the input via aria-describedby', () => {
    const html = renderToStaticMarkup(
      createElement(Toggle, {
        label: 'Show noise',
        checked: false,
        onChange: vi.fn(),
        hint: 'Adds Gaussian noise to the samples.',
      }),
    );
    const match = html.match(/aria-describedby="([^"]+)"/);
    expect(match).not.toBeNull();
    const describedById = match?.[1] ?? '';
    expect(html).toContain(`id="${describedById}"`);
    expect(html).toContain('Adds Gaussian noise to the samples.');
  });

  it('disables the input when disabled', () => {
    const html = renderToStaticMarkup(
      createElement(Toggle, { label: 'Show noise', checked: false, onChange: vi.fn(), disabled: true }),
    );
    expect(html).toContain('disabled=""');
  });
});
