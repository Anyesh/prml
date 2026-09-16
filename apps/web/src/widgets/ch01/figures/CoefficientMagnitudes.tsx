import {
  designMatrix,
  maximumLikelihoodWeights,
  pcg32,
  polynomialBasis,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260101;
const NOISE_STD = 0.2;
const N = 10;
const M_VALUES = [0, 1, 3, 6, 9];
const FLOOR = 1e-6;

const rng = pcg32(SEED, 1);
const XS = Array.from({ length: N }, (_, i) => i / (N - 1));
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));

function log10AbsCoefficients(m: number): number[] {
  const basis = polynomialBasis(m);
  const weights = maximumLikelihoodWeights(designMatrix(XS, basis), TS);
  return weights.map((w) => Math.log10(Math.max(Math.abs(w), FLOOR)));
}

const SERIES = M_VALUES.map((m) => ({ m, values: log10AbsCoefficients(m) }));
const MAX_LOG = Math.max(...SERIES.flatMap((s) => s.values));

export default function CoefficientMagnitudes() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={240} xDomain={[0, 9]} yDomain={[-1, Math.ceil(MAX_LOG) + 0.5]} label="Log magnitude of each fitted coefficient, for several polynomial orders">
        <Axes x={{ label: 'coefficient index j' }} y={{ label: 'log10 |w_j|' }} grid zeroLine />
        {SERIES.map((s, i) => (
          <Curve key={s.m} points={s.values.map((v, j) => [j, v] as const)} color={tokens.series[i % tokens.series.length]!} width={2} />
        ))}
        {SERIES.map((s, i) => (
          <ScatterField
            key={s.m}
            points={s.values.map((v, j) => ({ x: j, y: v, id: `${s.m}-${j}` }))}
            color={tokens.series[i % tokens.series.length]!}
            size={3.5}
          />
        ))}
        <Legend entries={SERIES.map((s, i) => ({ label: `M = ${s.m}`, color: tokens.series[i % tokens.series.length]!, mark: 'line' }))} placement="top-left" />
      </Plot>
      <p className="widget-readout">
        {`Largest |w_j| by order: ${SERIES.map((s) => `M=${s.m}: ${(10 ** Math.max(...s.values)).toExponential(2)}`).join(', ')}. Order jumps by three or four decades once M passes the point where the fit starts interpolating noise.`}
      </p>
    </div>
  );
}
