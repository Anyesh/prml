import type { ReactNode } from 'react';
import './Recall.css';

export interface RecallProps {
  question: string;
  children: ReactNode;
  /** Concept slugs this card exercises, read back by the review-deck compiler. */
  concepts?: readonly string[];
}

/**
 * Built on `<details>` rather than a React state toggle because recall cards sit in prose
 * pages that ship no JavaScript at all. A hydrated button would either force React onto
 * every section page or, unhydrated, render a toggle that does nothing.
 *
 * The answer stays in the DOM when collapsed, so the deck compiler and full-text search
 * both read it.
 */
export function Recall({ question, children, concepts }: RecallProps) {
  return (
    <details
      className="prml-recall"
      data-recall-question={question}
      data-recall-concepts={concepts?.join(',') ?? ''}
      data-pagefind-filter="type:recall"
    >
      <summary className="prml-recall-question">
        <span className="prml-recall-text">{question}</span>
        <span className="prml-recall-cue" aria-hidden="true">
          reveal
        </span>
      </summary>
      <div className="prml-recall-answer">{children}</div>
    </details>
  );
}
