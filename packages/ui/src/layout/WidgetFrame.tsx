import { useId, type ReactNode } from 'react';
import './WidgetFrame.css';

export interface WidgetFrameProps {
  title: string;
  caption: string;
  figure?: string;
  controls?: ReactNode;
  children: ReactNode;
  onReset?: () => void;
}

export function WidgetFrame({ title, caption, figure, controls, children, onReset }: WidgetFrameProps) {
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
        {onReset ? (
          <button type="button" className="prml-widget-frame-reset" onClick={onReset}>
            Reset
          </button>
        ) : null}
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
