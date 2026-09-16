import { useCallback, useState, type ReactNode } from 'react';
import './MathBlock.css';

export interface MathBlockProps {
  id?: string;
  name?: string;
  children: ReactNode;
  display?: boolean;
}

export function MathBlock({ id, name, children, display = true }: MathBlockProps) {
  const [copied, setCopied] = useState(false);
  const Tag = display ? 'div' : 'span';

  const copyLink = useCallback(() => {
    if (!id || typeof window === 'undefined' || !navigator.clipboard) return;
    const url = `${window.location.origin}${window.location.pathname}#${id}`;
    navigator.clipboard
      .writeText(url)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {
        setCopied(false);
      });
  }, [id]);

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
          <span className="prml-math-block-tag">({id})</span>
          <button
            type="button"
            className="prml-math-block-copy"
            onClick={copyLink}
            aria-label={copied ? 'Equation link copied' : `Copy link to equation ${id}`}
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
