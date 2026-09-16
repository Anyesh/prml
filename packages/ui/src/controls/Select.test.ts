import { describe, expect, it, vi } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Select } from './Select.js';

const KERNELS = [
  { value: 'rbf', label: 'RBF', hint: 'Smooth, infinite-dimensional feature map.' },
  { value: 'poly', label: 'Polynomial' },
  { value: 'linear', label: 'Linear' },
] as const;

describe('Select markup', () => {
  it('renders every option with its label', () => {
    const html = renderToStaticMarkup(
      createElement(Select, { label: 'Kernel', value: 'rbf', options: KERNELS, onChange: vi.fn() }),
    );
    expect(html).toContain('Kernel');
    expect(html).toContain('RBF');
    expect(html).toContain('Polynomial');
    expect(html).toContain('Linear');
  });

  it('carries a per-option hint as a title attribute', () => {
    const html = renderToStaticMarkup(
      createElement(Select, { label: 'Kernel', value: 'rbf', options: KERNELS, onChange: vi.fn() }),
    );
    expect(html).toContain('title="Smooth, infinite-dimensional feature map."');
  });

  it('wires the field hint via aria-describedby', () => {
    const html = renderToStaticMarkup(
      createElement(Select, {
        label: 'Kernel',
        value: 'rbf',
        options: KERNELS,
        onChange: vi.fn(),
        hint: 'Controls the similarity function.',
      }),
    );
    const match = html.match(/aria-describedby="([^"]+)"/);
    expect(match).not.toBeNull();
    expect(html).toContain('Controls the similarity function.');
  });

  it('disables the control when disabled', () => {
    const html = renderToStaticMarkup(
      createElement(Select, {
        label: 'Kernel',
        value: 'rbf',
        options: KERNELS,
        onChange: vi.fn(),
        disabled: true,
      }),
    );
    expect(html).toContain('disabled=""');
  });
});
