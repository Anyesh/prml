import { useId, type CSSProperties, type ReactNode } from 'react';
import './Panel.css';

export interface PanelProps {
  title?: string;
  children: ReactNode;
  columns?: 1 | 2 | 3;
  dense?: boolean;
}

export function Panel({ title, children, columns = 1, dense = false }: PanelProps) {
  const titleId = useId();

  return (
    <section
      className={`prml-panel${dense ? ' prml-panel--dense' : ''}`}
      aria-labelledby={title ? titleId : undefined}
      aria-label={title ? undefined : 'Controls'}
    >
      {title ? (
        <h3 id={titleId} className="prml-panel-title">
          {title}
        </h3>
      ) : null}
      <div
        className="prml-panel-grid"
        data-columns={columns}
        style={{ '--prml-panel-columns': columns } as CSSProperties}
      >
        {children}
      </div>
    </section>
  );
}
