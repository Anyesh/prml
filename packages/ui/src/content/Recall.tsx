import { useId, useState, type ReactNode } from 'react';
import './Recall.css';

export interface RecallProps {
  question: string;
  children: ReactNode;
  concepts?: readonly string[];
}

export function Recall({ question, children, concepts }: RecallProps) {
  const [revealed, setRevealed] = useState(false);
  const answerId = useId();

  return (
    <div
      className="prml-recall"
      data-recall-question={question}
      data-recall-concepts={concepts?.join(',') ?? ''}
      data-pagefind-filter="type:recall"
    >
      <p className="prml-recall-question">{question}</p>
      <button
        type="button"
        className="prml-recall-toggle"
        aria-expanded={revealed}
        aria-controls={answerId}
        onClick={() => setRevealed((r) => !r)}
      >
        {revealed ? 'Hide answer' : 'Show answer'}
      </button>
      {/* Stays mounted and merely hidden, both attribute and CSS-visible states, so a static-HTML
          crawler (the deck compiler, full-text search) can still read the answer text. */}
      <div id={answerId} className="prml-recall-answer" hidden={!revealed}>
        {children}
      </div>
    </div>
  );
}
