import { discreteEntropy, logGamma } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const RATIOS = [0.5, 0.3, 0.2];

function logMultiplicity(counts: number[]): number {
  const n = counts.reduce((s, c) => s + c, 0);
  return logGamma(n + 1) - counts.reduce((s, c) => s + logGamma(c + 1), 0);
}

const N_VALUES = [3, 10, 30, 100, 300, 1000, 3000];
const CURVE = N_VALUES.map((n) => {
  const counts = RATIOS.map((r) => Math.round(r * n));
  const total = counts.reduce((s, c) => s + c, 0);
  const logW = logMultiplicity(counts);
  return { n: total, perObject: logW / total };
});
const TARGET_H = discreteEntropy(RATIOS);

export default function MultiplicityAndEntropy() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 3000]} yDomain={[0.9, 1.15]} label="log(multiplicity) per object, against N, converging to the entropy of the fixed ratios">
        <Axes x={{ label: 'N' }} y={{ label: '(1/N) ln W' }} grid />
        <Curve points={CURVE.map((c) => [c.n, TARGET_H] as const)} color={tokens.color.inkFaint} dash="dashed" width={1.5} />
        <Curve points={CURVE.map((c) => [c.n, c.perObject] as const)} color={tokens.color.accent} width={2} />
        <Legend entries={[{ label: `H(0.5, 0.3, 0.2) = ${TARGET_H.toFixed(4)}`, color: tokens.color.inkFaint, mark: 'dashed-line' }, { label: '(1/N) ln W, exact', color: tokens.color.accent, mark: 'line' }]} placement="bottom-right" />
      </Plot>
      <p className="widget-readout">
        {`At N=${CURVE[0]!.n}: (1/N) ln W = ${CURVE[0]!.perObject.toFixed(4)}. By N=${CURVE[CURVE.length - 1]!.n}: ${CURVE[CURVE.length - 1]!.perObject.toFixed(4)}, within ${Math.abs(CURVE[CURVE.length - 1]!.perObject - TARGET_H).toExponential(2)} of the entropy 1.98 predicts directly from the ratios.`}
      </p>
    </div>
  );
}
