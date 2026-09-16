import { Annotation, Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const RATIO = 1.01;
const MAX_D = 1000;

function acceptanceCurve(): [number, number][] {
  return Array.from({ length: MAX_D }, (_, i) => {
    const d = i + 1;
    return [d, RATIO ** -d] as [number, number];
  });
}

export default function AcceptanceVsDimension() {
  const tokens = useResolvedTokens();
  const at1000 = RATIO ** -MAX_D;

  return (
    <div className="widget-grid">
      <Plot width={420} height={240} xDomain={[1, MAX_D]} yDomain={[1e-5, 1]} yScaleKind="log" label="Rejection-sampling acceptance rate collapsing with dimension">
        <Axes x={{ label: 'dimension D' }} y={{ label: 'acceptance rate (log scale)' }} grid />
        <Curve points={acceptanceCurve()} color={tokens.series[0]!} width={2.5} />
        <Annotation x={MAX_D} y={at1000} text={`D=1000: ${(at1000 * 100).toFixed(3)}%`} anchor="end" dy={14} plate />
      </Plot>
      <p className="widget-readout">
        A proposal only 1% wider than the target (σq = 1.01 σp) already accepts under 1 in 20,000 draws at D = 1000
        (PRML's own example): the exponential collapse in 11.1.2 is generic, not a property of a badly chosen envelope.
      </p>
    </div>
  );
}
