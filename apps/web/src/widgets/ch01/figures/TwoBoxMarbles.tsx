import { useMemo, useState } from 'react';
import { useResolvedTokens } from '@prml/viz';
import { Select } from '@prml/ui';
import '../../widgets.css';

const BOXES = {
  boxA: { apples: 2, oranges: 6, prior: 0.4 },
  boxB: { apples: 3, oranges: 1, prior: 0.6 },
} as const;

type BoxId = keyof typeof BOXES;
type FruitKind = 'apples' | 'oranges';

const BOX_IDS = Object.keys(BOXES) as BoxId[];

function likelihood(box: BoxId, fruit: FruitKind): number {
  const b = BOXES[box];
  return b[fruit] / (b.apples + b.oranges);
}

function posteriors(fruit: FruitKind): Record<BoxId, number> {
  const joints = BOX_IDS.map((box) => likelihood(box, fruit) * BOXES[box].prior);
  const total = joints.reduce((s, v) => s + v, 0);
  const out = {} as Record<BoxId, number>;
  BOX_IDS.forEach((box, i) => {
    out[box] = joints[i]! / total;
  });
  return out;
}

function fruitSquares(box: BoxId): FruitKind[] {
  const b = BOXES[box];
  return [...Array.from({ length: b.apples }, () => 'apples' as const), ...Array.from({ length: b.oranges }, () => 'oranges' as const)];
}

const BOX_LABELS: Record<BoxId, string> = { boxA: 'Box A', boxB: 'Box B' };

export default function TwoBoxMarbles() {
  const [fruit, setFruit] = useState<FruitKind>('oranges');
  const tokens = useResolvedTokens();

  const post = useMemo(() => posteriors(fruit), [fruit]);
  const pFruit = BOX_IDS.reduce((s, box) => s + likelihood(box, fruit) * BOXES[box].prior, 0);
  const fruitColor = (f: FruitKind) => (f === 'apples' ? tokens.color.accent : tokens.series[2]!);

  return (
    <div className="widget-grid">
      <div style={{ display: 'flex', gap: 'var(--prml-space-5)', flexWrap: 'wrap' }}>
        {BOX_IDS.map((box) => (
          <div key={box} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--prml-space-2)' }}>
            <span className="widget-readout">{`${BOX_LABELS[box]}, p(box) = ${BOXES[box].prior.toFixed(1)}`}</span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1.1rem)', gap: '0.2rem' }}>
              {fruitSquares(box).map((sq, i) => (
                <div
                  key={i}
                  style={{
                    width: '1.1rem',
                    height: '1.1rem',
                    borderRadius: '999px',
                    background: fruitColor(sq),
                    opacity: sq === fruit ? 1 : 0.35,
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <Select
        label="Observed fruit"
        value={fruit}
        onChange={(v) => setFruit(v as FruitKind)}
        options={[
          { value: 'apples', label: 'Apples' },
          { value: 'oranges', label: 'Oranges' },
        ]}
      />
      <p className="widget-readout">
        {`p(${fruit}) = ${pFruit.toFixed(3)}. Given ${fruit} was drawn: p(${BOX_LABELS.boxA}|${fruit}) = ${post.boxA.toFixed(3)}, p(${BOX_LABELS.boxB}|${fruit}) = ${post.boxB.toFixed(3)}.`}
      </p>
    </div>
  );
}
