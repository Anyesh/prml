import { describe, expect, it, vi } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { WidgetFrame } from './WidgetFrame.js';

describe('WidgetFrame markup', () => {
  it('renders the title, caption, and a noscript fallback', () => {
    const html = renderToStaticMarkup(
      createElement(WidgetFrame, {
        title: 'Bias-variance tradeoff',
        caption: 'Drag the slider to change model complexity.',
        children: createElement('div', null, 'plot'),
      }),
    );
    expect(html).toContain('Bias-variance tradeoff');
    expect(html).toContain('Drag the slider to change model complexity.');
    expect(html).toContain('<noscript>');
    expect(html).toContain('needs JavaScript');
  });

  it('shows the figure provenance note only when given', () => {
    const withFigure = renderToStaticMarkup(
      createElement(WidgetFrame, {
        title: 'T',
        caption: 'C',
        figure: '3.7',
        children: createElement('div', null, 'plot'),
      }),
    );
    expect(withFigure).toContain('Rebuilds Figure 3.7');

    const withoutFigure = renderToStaticMarkup(
      createElement(WidgetFrame, { title: 'T', caption: 'C', children: createElement('div', null, 'plot') }),
    );
    expect(withoutFigure).not.toContain('Rebuilds Figure');
  });

  it('carries no interactive control of its own, so the frame can render statically', () => {
    const html = renderToStaticMarkup(
      createElement(WidgetFrame, {
        title: 'T',
        caption: 'C',
        children: createElement('div', null, 'plot'),
      }),
    );
    // The widget nested inside is the hydrated island; a control here would be inert,
    // and reset belongs with the state it clears in any case.
    expect(html).not.toContain('<button');
  });

  it('places the controls panel inside the frame when given', () => {
    const html = renderToStaticMarkup(
      createElement(WidgetFrame, {
        title: 'T',
        caption: 'C',
        controls: createElement('div', null, 'the-controls'),
        children: createElement('div', null, 'plot'),
      }),
    );
    expect(html).toContain('the-controls');
  });
});
