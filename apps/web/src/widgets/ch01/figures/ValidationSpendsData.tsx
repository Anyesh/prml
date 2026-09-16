import { useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N = 20;

const SCHEMES = [
  { label: 'single hold-out (80/20)', trainFraction: 0.8, fits: 1 },
  { label: '5-fold cross-validation', trainFraction: 0.8, fits: 5 },
  { label: 'leave-one-out', trainFraction: (N - 1) / N, fits: N },
];

export default function ValidationSpendsData() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--prml-space-3)' }}>
        {SCHEMES.map((s) => (
          <div key={s.label} style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
            <span className="widget-readout">{`${s.label}: ${s.fits} model fit${s.fits > 1 ? 's' : ''}`}</span>
            <div style={{ display: 'flex', width: '100%', height: '1.1rem', borderRadius: 'var(--prml-radius-sm)', overflow: 'hidden' }}>
              <div style={{ width: `${s.trainFraction * 100}%`, background: tokens.color.accent }} />
              <div style={{ width: `${(1 - s.trainFraction) * 100}%`, background: tokens.color.danger }} />
            </div>
          </div>
        ))}
      </div>
      <p className="widget-readout">
        {`Blue trains, red validates, at N=${N}. Every scheme validates on all the data eventually; what differs is how many times the model gets refit to buy that, from 1 fit up to N.`}
      </p>
    </div>
  );
}
