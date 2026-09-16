import { conditionalEntropy, discreteEntropy, mutualInformation } from '@prml/math';
import { useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const JOINT = [
  [1 / 3, 1 / 3],
  [0, 1 / 3],
];
const P_X = [JOINT[0]![0]! + JOINT[0]![1]!, JOINT[1]![0]! + JOINT[1]![1]!];
const P_Y = [JOINT[0]![0]! + JOINT[1]![0]!, JOINT[0]![1]! + JOINT[1]![1]!];
const JOINT_TRANSPOSE = [
  [JOINT[0]![0]!, JOINT[1]![0]!],
  [JOINT[0]![1]!, JOINT[1]![1]!],
];

const H_X = discreteEntropy(P_X);
const H_Y = discreteEntropy(P_Y);
const H_XY = discreteEntropy(JOINT.flat());
const H_Y_GIVEN_X = conditionalEntropy(JOINT);
const H_X_GIVEN_Y = conditionalEntropy(JOINT_TRANSPOSE);
const I_XY = mutualInformation(JOINT);

const ROWS = [
  { label: 'H[x]', value: H_X },
  { label: 'H[y]', value: H_Y },
  { label: 'H[x,y]', value: H_XY },
  { label: 'H[y|x]', value: H_Y_GIVEN_X },
  { label: 'H[x|y]', value: H_X_GIVEN_Y },
  { label: 'I[x,y]', value: I_XY },
];

export default function MutualInformationDiagram() {
  const tokens = useResolvedTokens();
  const maxValue = Math.max(...ROWS.map((r) => r.value));

  return (
    <div className="widget-grid">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--prml-space-2)' }}>
        {ROWS.map((r) => (
          <div key={r.label} style={{ display: 'flex', alignItems: 'center', gap: 'var(--prml-space-3)' }}>
            <span className="widget-readout" style={{ width: '4.5rem' }}>{r.label}</span>
            <div style={{ width: '12rem', height: '0.9rem', background: 'var(--prml-color-border)', borderRadius: 'var(--prml-radius-sm)', overflow: 'hidden' }}>
              <div style={{ width: `${(r.value / maxValue) * 100}%`, height: '100%', background: tokens.color.accent }} />
            </div>
            <span className="widget-readout">{r.value.toFixed(4)}</span>
          </div>
        ))}
      </div>
      <p className="widget-readout">
        {`H[x,y] = ${H_XY.toFixed(4)} splits exactly into H[x] + H[y|x] = ${(H_X + H_Y_GIVEN_X).toFixed(4)}, per 1.112. I[x,y] = ${I_XY.toFixed(4)} is small but not zero: x and y are dependent, since x=1 rules out y=0 entirely.`}
      </p>
    </div>
  );
}
