import { describe, expect, it } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Recall } from './Recall.js';

describe('Recall markup', () => {
  it('emits the question and concepts as data attributes for the deck compiler', () => {
    const html = renderToStaticMarkup(
      createElement(Recall, {
        question: 'What does the KL divergence measure?',
        concepts: ['kl-divergence', 'entropy'],
        children: 'The information lost approximating one distribution with another.',
      }),
    );
    expect(html).toContain('data-recall-question="What does the KL divergence measure?"');
    expect(html).toContain('data-recall-concepts="kl-divergence,entropy"');
  });

  it('emits an empty concepts attribute rather than omitting it when none are given', () => {
    const html = renderToStaticMarkup(createElement(Recall, { question: 'Q', children: 'A' }));
    expect(html).toContain('data-recall-concepts=""');
  });

  it('carries the pagefind filter for the review deck', () => {
    const html = renderToStaticMarkup(createElement(Recall, { question: 'Q', children: 'A' }));
    expect(html).toContain('data-pagefind-filter="type:recall"');
  });

  it('keeps the answer in the DOM but collapsed before reveal', () => {
    const html = renderToStaticMarkup(
      createElement(Recall, { question: 'Q', children: 'The hidden answer text' }),
    );
    expect(html).toContain('The hidden answer text');
    expect(html).not.toContain('open=""');
  });

  it('reveals through native details, so a page shipping no JavaScript still works', () => {
    const html = renderToStaticMarkup(createElement(Recall, { question: 'Q', children: 'A' }));
    expect(html).toMatch(/^<details/);
    expect(html).toContain('<summary');
    expect(html).not.toContain('<button');
  });
});
