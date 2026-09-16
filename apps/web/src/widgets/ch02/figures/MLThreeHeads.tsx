import { betaMean, betaPosterior } from '@prml/math';
import { useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const PRIOR = { a: 2, b: 2 };
const HEADS = 3;
const TAILS = 0;
const ML_ESTIMATE = HEADS / (HEADS + TAILS);
const POSTERIOR = betaPosterior(PRIOR, HEADS, TAILS);
const BAYES_PREDICTIVE = betaMean(POSTERIOR);

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '9rem 1fr 3rem', alignItems: 'center', gap: 'var(--prml-space-2)' }}>
      <span className="widget-readout">{label}</span>
      <div style={{ background: 'var(--prml-color-sunken)', borderRadius: 'var(--prml-radius-sm)', height: '0.9rem' }}>
        <div style={{ width: `${value * 100}%`, height: '100%', background: color, borderRadius: 'var(--prml-radius-sm)' }} />
      </div>
      <span className="widget-readout">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

export default function MLThreeHeads() {
  const tokens = useResolvedTokens();
  return (
    <div style={{ display: 'grid', gap: 'var(--prml-space-3)' }}>
      <Bar label="ML estimate" value={ML_ESTIMATE} color={tokens.color.danger} />
      <Bar label="Bayesian predictive" value={BAYES_PREDICTIVE} color={tokens.color.accent} />
      <p className="widget-readout">
        {`Three flips, three heads. ML gives µ = ${ML_ESTIMATE.toFixed(2)}: certainty of heads forever. `}
        {`A Beta(2, 2) prior instead gives predictive p(x=1) = ${BAYES_PREDICTIVE.toFixed(3)} (eq. 2.20), pulled back from 1 `}
        {'by two fictitious prior observations of each outcome.'}
      </p>
    </div>
  );
}
