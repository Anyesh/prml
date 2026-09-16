import { useId, type ReactNode } from 'react';
import './WidgetFrame.css';

export interface WidgetFrameProps {
  title: string;
  /** One sentence telling the reader what to *do* with this widget, not what it shows. */
  caption: string;
  /** Book figure this rebuilds, e.g. "3.7". */
  figure?: string;
  controls?: ReactNode;
  children: ReactNode;
}

/**
 * Presentational chrome only, rendered statically by the page. The interactive island is
 * the widget nested inside it, which is why there is no reset control here: hydrating the
 * frame would leave the widget itself static, and a reset button belongs with the state it
 * clears in any case.
 */
export function WidgetFrame({ title, caption, figure, controls, children }: WidgetFrameProps) {
  const titleId = useId();

  return (
    <figure className="prml-widget-frame" aria-labelledby={titleId}>
      <header className="prml-widget-frame-header">
        <div className="prml-widget-frame-heading">
          <h3 id={titleId} className="prml-widget-frame-title">
            {title}
          </h3>
          {figure ? <p className="prml-widget-frame-provenance">Rebuilds Figure {figure}</p> : null}
        </div>
      </header>
      <figcaption className="prml-widget-frame-caption">{caption}</figcaption>
      <noscript>
        <p className="prml-widget-frame-noscript">
          This figure is interactive and needs JavaScript. Turn on JavaScript to explore it here, or
          read the surrounding text for the same idea worked through in prose.
        </p>
      </noscript>
      <div className="prml-widget-frame-stage">{children}</div>
      {controls ? <div className="prml-widget-frame-controls">{controls}</div> : null}
    </figure>
  );
}
