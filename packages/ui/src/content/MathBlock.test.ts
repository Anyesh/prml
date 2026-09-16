import { describe, expect, it } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MathBlock } from './MathBlock.js';

describe('MathBlock markup', () => {
  it('wraps already-rendered math without touching it', () => {
    const html = renderToStaticMarkup(
      createElement(MathBlock, {
        id: '3.49',
        children: createElement('span', { className: 'katex' }, 'rendered math'),
      }),
    );
    expect(html).toContain('class="katex"');
    expect(html).toContain('rendered math');
  });

  it('sets the anchor id and the searchable equation metadata', () => {
    const html = renderToStaticMarkup(createElement(MathBlock, { id: '3.49', children: 'x' }));
    expect(html).toContain('id="3.49"');
    expect(html).toContain('data-pagefind-meta="equation:3.49"');
    expect(html).toContain('(3.49)');
  });

  it('omits the tag, copy button, and pagefind metadata without an id', () => {
    const html = renderToStaticMarkup(createElement(MathBlock, { children: 'x' }));
    expect(html).not.toContain('data-pagefind-meta');
    expect(html).not.toContain('Copy link');
  });

  it('shows the name as a hover title', () => {
    const html = renderToStaticMarkup(
      createElement(MathBlock, { id: '3.49', name: 'posterior mean', children: 'x' }),
    );
    expect(html).toContain('title="posterior mean"');
  });

  it('renders inline math as a span rather than a div', () => {
    const html = renderToStaticMarkup(createElement(MathBlock, { display: false, children: 'x' }));
    expect(html).toMatch(/^<span/);
  });

  it('renders display math as a div by default', () => {
    const html = renderToStaticMarkup(createElement(MathBlock, { children: 'x' }));
    expect(html).toMatch(/^<div/);
  });
});
