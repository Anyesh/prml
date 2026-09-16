import { dirichletMean } from '@prml/math';
import { useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const PRIOR_ALPHA = [2, 2, 2];
const COUNTS = [5, 2, 9];
const POSTERIOR_ALPHA = PRIOR_ALPHA.map((a, k) => a + COUNTS[k]!);
const PRIOR_MEAN = dirichletMean({ alpha: PRIOR_ALPHA });
const POSTERIOR_MEAN = dirichletMean({ alpha: POSTERIOR_ALPHA });
const LABELS = ['A', 'B', 'C'];

export default function MultinomialPosteriorUpdate() {
  const tokens = useResolvedTokens();
  return (
    <div style={{ display: 'grid', gap: 'var(--prml-space-3)' }}>
      {LABELS.map((label, k) => (
        <div key={label} style={{ display: 'grid', gridTemplateColumns: '2rem 1fr 1fr', alignItems: 'center', gap: 'var(--prml-space-2)' }}>
          <span className="widget-readout">{label}</span>
          <div style={{ background: 'var(--prml-color-sunken)', borderRadius: 'var(--prml-radius-sm)', height: '0.8rem' }}>
            <div
              style={{
                width: `${PRIOR_MEAN[k]! * 100}%`,
                height: '100%',
                background: tokens.color.inkFaint,
                borderRadius: 'var(--prml-radius-sm)',
              }}
            />
          </div>
          <div style={{ background: 'var(--prml-color-sunken)', borderRadius: 'var(--prml-radius-sm)', height: '0.8rem' }}>
            <div
              style={{
                width: `${POSTERIOR_MEAN[k]! * 100}%`,
                height: '100%',
                background: tokens.color.accent,
                borderRadius: 'var(--prml-radius-sm)',
              }}
            />
          </div>
        </div>
      ))}
      <p className="widget-readout">
        {`Counts (${COUNTS.join(', ')}) turn a uniform Dirichlet(2,2,2) prior (grey) into posterior `}
        {`Dirichlet(${POSTERIOR_ALPHA.join(', ')}) (blue, eq. 2.41), mean shifting from `}
        {`(${PRIOR_MEAN.map((v) => v.toFixed(2)).join(', ')}) toward the observed proportions.`}
      </p>
    </div>
  );
}
