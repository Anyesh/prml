import { describe, expect, it } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Panel } from './Panel.js';

describe('Panel markup', () => {
  it('labels the section from its title when given', () => {
    const html = renderToStaticMarkup(
      createElement(Panel, { title: 'Prior', children: createElement('span', null, 'child') }),
    );
    expect(html).toContain('Prior');
    expect(html).toMatch(/aria-labelledby="[^"]+"/);
  });

  it('falls back to a generic label when no title is given', () => {
    const html = renderToStaticMarkup(
      createElement(Panel, { children: createElement('span', null, 'child') }),
    );
    expect(html).toContain('aria-label="Controls"');
  });

  it('defaults to a single-column grid', () => {
    const html = renderToStaticMarkup(
      createElement(Panel, { children: createElement('span', null, 'child') }),
    );
    expect(html).toContain('data-columns="1"');
    expect(html).toContain('--prml-panel-columns:1');
  });

  it('carries the requested column count so the container query can narrow it', () => {
    for (const columns of [2, 3] as const) {
      const html = renderToStaticMarkup(
        createElement(Panel, { columns, children: createElement('span', null, 'child') }),
      );
      expect(html).toContain(`data-columns="${columns}"`);
      expect(html).toContain(`--prml-panel-columns:${columns}`);
    }
  });
});
