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
    expect(html).toContain('prml-panel-grid--1');
  });

  it('accepts a two-column layout', () => {
    const html = renderToStaticMarkup(
      createElement(Panel, { columns: 2, children: createElement('span', null, 'child') }),
    );
    expect(html).toContain('prml-panel-grid--2');
  });
});
