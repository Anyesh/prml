import { pcg32 } from '@prml/math';
import { useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const X_STATES = 9;
const Y_STATES = 2;

function buildJoint(): number[][] {
  const rng = pcg32(20260601);
  const raw = Array.from({ length: Y_STATES }, () => Array.from({ length: X_STATES }, () => rng.next() ** 2));
  const total = raw.flat().reduce((s, v) => s + v, 0);
  return raw.map((row) => row.map((v) => v / total));
}

const JOINT = buildJoint();
const P_X = Array.from({ length: X_STATES }, (_, i) => JOINT[0]![i]! + JOINT[1]![i]!);
const P_Y = [JOINT[0]!.reduce((s, v) => s + v, 0), JOINT[1]!.reduce((s, v) => s + v, 0)];
const P_X_GIVEN_Y1 = P_X.map((_, i) => JOINT[1]![i]! / P_Y[1]!);
const MAX_CELL = Math.max(...JOINT.flat());

export default function JointMarginalConditional() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--prml-space-2)' }}>
        <span className="widget-readout">joint p(X, Y), darker means more probable</span>
        {JOINT.map((row, j) => (
          <div key={j} style={{ display: 'grid', gridTemplateColumns: `repeat(${X_STATES}, 1.4rem)`, gap: '0.15rem' }}>
            {row.map((p, i) => (
              <div
                key={i}
                style={{ width: '1.4rem', height: '1.4rem', background: tokens.color.accent, opacity: 0.15 + 0.85 * (p / MAX_CELL) }}
              />
            ))}
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--prml-space-2)' }}>
        <span className="widget-readout">marginal p(X), each bar summed down its column</span>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${X_STATES}, 1.4rem)`, gap: '0.15rem', alignItems: 'end', height: '3rem' }}>
          {P_X.map((p, i) => (
            <div key={i} style={{ width: '1.4rem', height: `${(p / Math.max(...P_X)) * 100}%`, background: tokens.series[1]! }} />
          ))}
        </div>
      </div>
      <p className="widget-readout">
        {`p(Y=1) = ${P_Y[1]!.toFixed(3)}. Conditioning on Y=1 renormalises just that row: p(X|Y=1) sums to 1 over the ${X_STATES} states, with the largest at x=${P_X_GIVEN_Y1.indexOf(Math.max(...P_X_GIVEN_Y1)) + 1} (p=${Math.max(...P_X_GIVEN_Y1).toFixed(3)}).`}
      </p>
    </div>
  );
}
