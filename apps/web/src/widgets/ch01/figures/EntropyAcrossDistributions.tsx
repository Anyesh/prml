import { discreteEntropy } from '@prml/math';
import { useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DISTRIBUTIONS = [
  { label: 'one-hot', probs: [1, 0, 0, 0, 0, 0, 0, 0] },
  { label: "PRML's skewed 8-state code", probs: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 64, 1 / 64, 1 / 64, 1 / 64] },
  { label: 'uniform, 8 states', probs: Array(8).fill(1 / 8) },
];

const ENTROPIES = DISTRIBUTIONS.map((d) => discreteEntropy(d.probs) / Math.LN2);
const MAX_H = Math.log2(8);

export default function EntropyAcrossDistributions() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--prml-space-3)' }}>
        {DISTRIBUTIONS.map((d, i) => (
          <div key={d.label} style={{ display: 'flex', alignItems: 'center', gap: 'var(--prml-space-3)' }}>
            <span className="widget-readout" style={{ width: '13rem' }}>{d.label}</span>
            <div style={{ width: '14rem', height: '0.9rem', background: 'var(--prml-color-border)', borderRadius: 'var(--prml-radius-sm)', overflow: 'hidden' }}>
              <div style={{ width: `${(ENTROPIES[i]! / MAX_H) * 100}%`, height: '100%', background: tokens.color.accent }} />
            </div>
            <span className="widget-readout">{`${ENTROPIES[i]!.toFixed(3)} bits`}</span>
          </div>
        ))}
      </div>
      <p className="widget-readout">
        {`Over 8 states the ceiling is log2(8) = ${MAX_H.toFixed(3)} bits, reached only by the uniform row. Every state that gets more probable than 1/8 somewhere else gets less probable, and entropy can only fall.`}
      </p>
    </div>
  );
}
