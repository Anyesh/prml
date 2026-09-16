import type { ReactNode } from 'react';
import './MathBlock.css';

export interface MathBlockProps {
  /** The book's equation number. Doubles as the anchor id, so links read `#3.49`. */
  id?: string;
  name?: string;
  children: ReactNode;
  display?: boolean;
}

/**
 * Wraps math that `rehype-katex` already rendered to HTML at build time. It must never
 * render math itself, because doing so would put KaTeX in the client bundle of pages that
 * otherwise ship nothing.
 *
 * The copy-link control carries a `data-copy-anchor` attribute rather than an `onClick`,
 * and a delegated listener in the site layout handles it. Equations live in prose pages
 * with no React on them, so a hydrated handler would either drag the framework onto every
 * section page or render a button that silently does nothing.
 */
export function MathBlock({ id, name, children, display = true }: MathBlockProps) {
  const Tag = display ? 'div' : 'span';

  return (
    <Tag
      id={id}
      className={`prml-math-block ${display ? 'prml-math-block--display' : 'prml-math-block--inline'}`}
      data-pagefind-meta={id ? `equation:${id}` : undefined}
      title={name}
    >
      <span className="prml-math-block-content">{children}</span>
      {id ? (
        <span className="prml-math-block-tag-group">
          <a className="prml-math-block-tag" href={`#${id}`}>
            ({id})
          </a>
          <button
            type="button"
            className="prml-math-block-copy"
            data-copy-anchor={id}
            aria-label={`Copy link to equation ${id}`}
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path
                d="M6.7 9.3a2 2 0 0 1 0-2.83l1.6-1.6a2 2 0 1 1 2.83 2.83l-.7.7M9.3 6.7a2 2 0 0 1 0 2.83l-1.6 1.6a2 2 0 1 1-2.83-2.83l.7-.7"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                fill="none"
              />
            </svg>
          </button>
        </span>
      ) : null}
    </Tag>
  );
}
