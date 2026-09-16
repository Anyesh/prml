import '../../widgets.css';

const K = 6;
const ACTIVE = 2;

export default function OneOfKCoding() {
  return (
    <div>
      <div style={{ display: 'flex', gap: 'var(--prml-space-2)' }}>
        {Array.from({ length: K }, (_, k) => (
          <div
            key={k}
            style={{
              width: '2.4rem',
              height: '2.4rem',
              display: 'grid',
              placeItems: 'center',
              borderRadius: 'var(--prml-radius-sm)',
              border: '1px solid var(--prml-color-border-strong)',
              background: k === ACTIVE ? 'var(--prml-color-accent)' : 'var(--prml-color-sunken)',
              color: k === ACTIVE ? 'var(--prml-color-bg)' : 'var(--prml-color-ink-muted)',
              fontFamily: 'var(--prml-font-mono, monospace)',
            }}
          >
            {k === ACTIVE ? 1 : 0}
          </div>
        ))}
      </div>
      <p className="widget-readout">
        {`x = (0,0,1,0,0,0)ᵀ codes "category 3 of 6" (eq. 2.25): one coordinate on, the rest off, exactly one 1 per observation.`}
      </p>
    </div>
  );
}
