import { useState } from 'react';
import { useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../../widgets.css';

const N = 20;
const S = 4;

export default function FoldPartitionDiagram() {
  const [fold, setFold] = useState(0);
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--prml-space-2)' }}>
        {Array.from({ length: S }, (_, run) => (
          <div key={run} style={{ display: 'flex', alignItems: 'center', gap: 'var(--prml-space-2)' }}>
            <span className="widget-readout" style={{ width: '4.5rem' }}>{`run ${run + 1}`}</span>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${N}, 0.9rem)`, gap: '0.15rem' }}>
              {Array.from({ length: N }, (_, i) => (
                <div
                  key={i}
                  style={{
                    width: '0.9rem',
                    height: '0.9rem',
                    background: i % S === run ? tokens.color.danger : tokens.color.accent,
                    opacity: run === fold ? 1 : 0.3,
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <StepThrough step={fold} stepCount={S} onStep={setFold} labels={['run 1', 'run 2', 'run 3', 'run 4']} />
      <p className="widget-readout">
        {`Red is held out, blue trains. Step through the four runs: every point is red exactly once, so all ${N} points get scored while ${S - 1}/${S} of the data trains each run.`}
      </p>
    </div>
  );
}
