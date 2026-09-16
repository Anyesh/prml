import { useId, type ReactNode } from 'react';
import './Figure.css';

export interface FigureProps {
  /** One line naming what the reader should see. Not a title: a claim the picture supports. */
  caption: string;
  /** Book figure this rebuilds, e.g. "3.6". Omit when the figure is ours rather than the book's. */
  source?: string;
  /** Sits in the prose column by default; `wide` breaks out for anything with two panels. */
  width?: 'column' | 'wide';
  children: ReactNode;
}

/**
 * Lightweight chrome for a figure that sits inline in the argument, as opposed to the one
 * full exploration a section builds towards, which uses `WidgetFrame`.
 *
 * Deliberately quieter than `WidgetFrame`: no heading, no provenance row, no border by
 * default. A section carries several of these, and giving each the full apparatus would
 * turn the page into a stack of boxes and bury the prose they are explaining.
 */
export function Figure({ caption, source, width = 'column', children }: FigureProps) {
  const captionId = useId();

  return (
    <figure className={`prml-figure prml-figure--${width}`} aria-describedby={captionId}>
      <div className="prml-figure-stage">{children}</div>
      <figcaption id={captionId} className="prml-figure-caption">
        {caption}
        {source ? <span className="prml-figure-source">Figure {source}</span> : null}
      </figcaption>
    </figure>
  );
}
